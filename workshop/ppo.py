"""One token-level PPO implementation for all reward arms.

Rollout statistics and GAE are computed once, then consumed directly by the
optimizer. This retains the original clipped policy/value objectives, frozen
LoRA-disabled reference, response masking, optimizer settings and random seeds.
"""
from pathlib import Path
import os
import time

import numpy as np
import torch

from .common import ENGINE_ID, atomic_json, file_sha, read_json, seed_for


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


def advantages_and_returns(values, rewards, mask, gamma, lam):
    advantages = torch.zeros_like(values)
    running = torch.zeros(values.shape[0], device=values.device)
    for t in range(values.shape[1]-1, -1, -1):
        alive = mask[:, t+1].float() if t+1 < values.shape[1] else torch.zeros_like(running)
        next_value = values[:, t+1] if t+1 < values.shape[1] else torch.zeros_like(running)
        delta = rewards[:, t] + gamma * next_value * alive - values[:, t]
        running = (delta + gamma * lam * running * alive) * mask[:, t]
        advantages[:, t] = running
    returns = (advantages + values) * mask
    valid = advantages[mask]
    normalized = (advantages - valid.mean()) / torch.sqrt(valid.var(unbiased=False) + 1e-8)
    return normalized * mask, returns


class PPOTrainer:
    def __init__(self, policy, config):
        self.policy, self.config, self.c = policy, config, config['ppo']
        self.microbatch = config['runtime']['ppo_microbatch_size']
        self.optimizer = torch.optim.AdamW([
            {'params': [p for p in policy.lm.parameters() if p.requires_grad], 'lr': self.c['learning_rate']},
            {'params': policy.value_head.parameters(), 'lr': self.c['value_learning_rate']}],
            betas=(.9, .999), eps=1e-5, weight_decay=0.)

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
        kl = (old - refs) * mask
        token_rewards = -c['kl_coefficient'] * kl
        token_rewards[torch.arange(len(items), device=policy.device), mask.sum(1)-1] += torch.as_tensor(rewards, dtype=torch.float32, device=policy.device)
        advantages, returns = advantages_and_returns(values, token_rewards, mask, c['gamma'], c['gae_lambda'])
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
        """Consume the snapshot; never recompute old/reference statistics or GAE."""
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
                self.optimizer.zero_grad(set_to_none=True)
                accum = dict(policy_loss=0., value_loss=0., old_policy_kl=0., clip_fraction=0.)
                for offset in range(0, len(mini), self.microbatch):
                    ix = mini[offset:offset+self.microbatch]
                    newp, values = actor.statistics(ids[ix], attention[ix], rollout['prompt_width'])
                    log_ratio = newp - old[ix]
                    if not torch.isfinite(log_ratio[mask[ix]]).all() or log_ratio[mask[ix]].abs().max() > 20:
                        raise FloatingPointError('Nonfinite or extreme PPO likelihood ratio.')
                    ratio = log_ratio.exp()
                    surrogate = torch.minimum(ratio*adv[ix], ratio.clamp(1-c['clip_range'], 1+c['clip_range'])*adv[ix])
                    clipped_v = old_values[ix] + (values-old_values[ix]).clamp(-c['value_clip_range'], c['value_clip_range'])
                    vloss = .5 * torch.maximum((values-returns[ix]).square(), (clipped_v-returns[ix]).square())
                    p_loss = -(surrogate*mask[ix]).sum()/denominator
                    v_loss = (vloss*mask[ix]).sum()/denominator
                    loss = p_loss + c['value_coefficient']*v_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite PPO loss.')
                    loss.backward()
                    accum['policy_loss'] += p_loss.detach().item()
                    accum['value_loss'] += v_loss.detach().item()
                    accum['old_policy_kl'] += (((ratio-1)-log_ratio)*mask[ix]).sum().detach().item()/denominator.item()
                    accum['clip_fraction'] += (((ratio-1).abs()>c['clip_range'])*mask[ix]).sum().item()/denominator.item()
                if accum['old_policy_kl'] > c['target_update_kl']:
                    self.optimizer.zero_grad(set_to_none=True)
                    stopped_early = True
                    break
                norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), c['max_grad_norm'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite gradient norm.')
                self.optimizer.step()
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
