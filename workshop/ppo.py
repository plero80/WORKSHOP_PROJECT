"""Experiment batching/checkpoints around OpenRLHF's PPO training steps.

All reward arms use the same pinned library backend. The project supplies
tokens/rewards and persists its shared actor/value parameters and optimizer.
"""
from pathlib import Path
import os
import time

import numpy as np
import torch

from .common import ENGINE_ID, atomic_json, file_sha, read_json, seed_for
from .openrlhf_backend import build_workers


def pack_items(items, pad_id):
    if not items or any(not x['prompt_ids'] or not x['response_ids'] for x in items):
        raise ValueError('Each rollout needs nonempty prompts and responses.')
    width = max(len(x['prompt_ids']) for x in items)
    responses = max(len(x['response_ids']) for x in items)
    ids = torch.full((len(items), width + responses), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    mask = torch.zeros((len(items), responses), dtype=torch.bool)
    for i, item in enumerate(items):
        prompt, response = item['prompt_ids'], item['response_ids']
        ids[i, width-len(prompt):width+len(response)] = torch.tensor(prompt+response)
        attention[i, width-len(prompt):width+len(response)] = 1
        mask[i, :len(response)] = True
    return {'ids': ids, 'attention': attention, 'mask': mask, 'prompt_width': width}


class PPOTrainer:
    def __init__(self, policy, config):
        self.policy, self.config, self.c = policy, config, config['ppo']
        self.microbatch = config['runtime']['ppo_microbatch_size']
        fused = config['runtime'].get('fused_optimizer', False) and policy.device.type == 'cuda'
        self.optimizer = torch.optim.AdamW([
            {'params': [p for p in policy.lm.parameters() if p.requires_grad], 'lr': self.c['learning_rate']},
            {'params': policy.value_head.parameters(), 'lr': self.c['value_learning_rate']}],
            betas=(.9, .999), eps=1e-5, weight_decay=0., fused=fused)
        self.workers = build_workers(policy, self.optimizer, self.c)

    @torch.no_grad()
    def prepare(self, items, rewards):
        """Snapshot the old policy/reference once for this already graded rollout."""
        if not items or len(items) != len(rewards) or not np.isfinite(rewards).all():
            raise ValueError('Need aligned nonempty items and finite rewards.')
        policy, c = self.policy, self.c
        policy.eval()
        batch = pack_items(items, policy.tokenizer.pad_token_id)
        batch = {k: v.to(policy.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        old, values, refs = [], [], []
        for start in range(0, len(items), self.microbatch):
            sl = slice(start, start+self.microbatch)
            args = batch['ids'][sl], batch['attention'][sl], batch['prompt_width']
            lp, val = policy.statistics(*args)
            ref, _ = policy.statistics(*args, reference=True, with_values=False)
            old.append(lp); values.append(val); refs.append(ref)
        old, values, refs = map(torch.cat, (old, values, refs))
        mask = batch['mask']
        lib = self.workers.library
        kl = lib.compute_approx_kl(old, refs, kl_estimator='k1') * mask
        scalar_rewards = torch.as_tensor(rewards, dtype=torch.float32, device=policy.device)
        token_rewards = lib.compute_reward(scalar_rewards, c['kl_coefficient'], kl, action_mask=mask)
        experience = lib.Experience(index=list(range(len(items))), sequences=batch['ids'],
            attention_mask=batch['attention'], action_mask=mask, action_log_probs=old,
            base_action_log_probs=refs, values=values, kl=kl, rewards=scalar_rewards,
            info={'reward': scalar_rewards, 'response_length': mask.sum(1).float(),
                  'total_length': batch['attention'].sum(1).float()})
        self.workers.maker.compute_advantages_and_returns([experience])
        advantages, returns = experience.advantages * mask, experience.returns * mask
        return {**batch, 'old_logprobs': old, 'old_values': values, 'reference_logprobs': refs,
                'advantages': advantages, 'returns': returns, 'kl': kl, 'token_rewards': token_rewards}

    def update(self, items, rewards, update_index):
        start = time.monotonic()
        rollout = self.prepare(items, rewards)
        prepared = time.monotonic()
        result = self.optimize(rollout, update_index)
        return {**result, 'rollout_stats_seconds': prepared-start,
                'optimization_seconds': time.monotonic()-prepared}

    def optimize(self, rollout, update_index):
        """Feed frozen rollout statistics to the library's actor/critic steps."""
        c, actor = self.c, self.policy
        ids, attention, mask = (rollout[k] for k in ('ids', 'attention', 'mask'))
        old, old_values, adv, returns = (rollout[k] for k in ('old_logprobs', 'old_values', 'advantages', 'returns'))
        if update_index == 0 and rollout['kl'].abs().max().item() > .05:
            raise RuntimeError('Initial LoRA policy should match the frozen reference.')
        metrics, stopped_early = [], False
        for epoch in range(c['epochs']):
            order = np.random.default_rng(seed_for(self.config['seed'], update_index+1, epoch, 'minibatches')).permutation(len(ids))
            for start in range(0, len(ids), c['minibatch_size']):
                mini = order[start:start+c['minibatch_size']]
                denominator = mask[mini].sum().clamp_min(1)
                self.workers.strategy.begin()
                accum = dict(policy_loss=0., value_loss=0., old_policy_kl=0., clip_fraction=0.)
                microbatches = 0
                for offset in range(0, len(mini), self.microbatch):
                    ix = mini[offset:offset+self.microbatch]
                    scale = (mask[ix].sum() / denominator).item()
                    self.workers.strategy.scale = scale
                    self.workers.actor.actor.old_logprobs = old[ix]
                    experience = self.workers.library.Experience(
                        sequences=ids[ix], attention_mask=attention[ix], action_mask=mask[ix],
                        action_log_probs=old[ix], base_action_log_probs=rollout['reference_logprobs'][ix],
                        values=old_values[ix], advantages=adv[ix], returns=returns[ix],
                        info={'response_length': mask[ix].sum(1).float()})
                    actor_stats = self.workers.actor.training_step(experience, c['kl_coefficient'], offset)
                    critic_stats = self.workers.critic.training_step(experience, offset)
                    accum['policy_loss'] += actor_stats['policy_loss'] * scale
                    accum['value_loss'] += critic_stats['critic_loss'] * scale
                    accum['clip_fraction'] += actor_stats['ppo_clip_ratio'] * scale
                    newp = self.workers.actor.actor.last_logprobs
                    # k3 is a nonnegative update diagnostic, separate from the
                    # k1 frozen-reference penalty used in the reward.
                    update_kl = self.workers.library.compute_approx_kl(old[ix], newp, kl_estimator='k3')
                    accum['old_policy_kl'] += (update_kl * mask[ix]).sum().item() / denominator.item()
                    microbatches += 1
                if accum['old_policy_kl'] > c['target_update_kl']:
                    self.optimizer.zero_grad(set_to_none=True)
                    stopped_early = True
                    break
                self.workers.strategy.finish(microbatches)
                metrics.append(accum)
            if stopped_early:
                break
        if not metrics:
            raise RuntimeError('PPO completed no optimizer step; inspect likelihood diagnostics.')
        self.optimizer.zero_grad(set_to_none=True)
        return {**{k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
                'optimizer_steps': len(metrics), 'early_stopped_on_update_kl': stopped_early,
                'sampled_reference_kl_per_response': rollout['kl'].sum(1).mean().item(),
                'rollout_response_tokens': int(mask.sum()), 'engine': ENGINE_ID}


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(v) for v in value)
    return value


def save_checkpoint(path, policy, optimizer, step, fingerprint, arm, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {'engine': ENGINE_ID, 'step': step, 'fingerprint': fingerprint, 'arm': arm,
             'trainable': policy.trainable_state(), 'optimizer': _cpu_tree(optimizer.state_dict()),
             'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             'extra': extra or {}}
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    atomic_json(path.with_suffix('.sha256.json'), {'sha256': file_sha(path), 'engine': ENGINE_ID})


def load_checkpoint(path, policy, optimizer, fingerprint, arm):
    path = Path(path)
    info = read_json(path.with_suffix('.sha256.json'))
    if info.get('engine') != ENGINE_ID or info.get('sha256') != file_sha(path):
        raise ValueError('Checkpoint checksum or engine mismatch.')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('engine') != ENGINE_ID or state['fingerprint'] != fingerprint or state['arm'] != arm:
        raise ValueError('Checkpoint belongs to another experiment or arm.')
    policy.restore_trainable(state['trainable'])
    optimizer.load_state_dict(state['optimizer'])
    for values in optimizer.state.values():
        for key, val in values.items():
            if isinstance(val, torch.Tensor) and key != 'step':
                values[key] = val.to(policy.device)
    torch.set_rng_state(state['torch_rng'].cpu())
    if state['cuda_rng'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    return state
