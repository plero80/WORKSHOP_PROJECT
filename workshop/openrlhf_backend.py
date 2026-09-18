"""Pinned OpenRLHF training steps, adapted to the shared single-device policy.

Only model/strategy initialization is specialized here. The actor and critic
training_step methods, clipped losses, reward shaping, GAE and advantage
normalization execute in the installed OpenRLHF package. No Ray cluster or vLLM
generation engine is started. See docs/OPENRLHF.md for the integration boundary.
"""
from functools import lru_cache
from importlib import metadata
import inspect
import os
from pathlib import Path
import sys
from types import SimpleNamespace

# Set before optional DeepSpeed imports, including those performed by Accelerate
# during HF model saves. Use the compiler installed by scripts/setup.sh if needed.
_compiler = Path(sys.prefix)/'cuda-toolkit'
if (_compiler/'bin/nvcc').is_file():
    os.environ.setdefault('CUDA_HOME', str(_compiler))

import torch
from torch import nn

VERSION = "0.9.0"
REVISION = "06ab51e75c0e8c5b4d6b66912996b2a611e3dc2f"


@lru_cache(maxsize=1)
def library():
    try:
        installed = metadata.version("openrlhf")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("OpenRLHF is required for training. On Linux, run bash scripts/setup.sh.") from error
    if installed != VERSION:
        raise RuntimeError(f"Expected OpenRLHF {VERSION}, found {installed}. Run bash scripts/setup.sh.")
    try:
        from openrlhf.models import PolicyLoss, ValueLoss
        from openrlhf.models.utils import compute_approx_kl, compute_reward
        from openrlhf.trainer.ppo_utils.experience_maker import Experience, RemoteExperienceMaker
        from openrlhf.trainer.ray.ppo_actor import ActorPPOTrainer
        from openrlhf.trainer.ray.ppo_critic import CriticPPOTrainer
    except (ImportError, OSError) as error:
        raise RuntimeError("The OpenRLHF training environment is incomplete or incompatible. "
                           "Use Linux/CUDA and run bash scripts/setup.sh. " + str(error)) from error
    return SimpleNamespace(**locals())


def provenance():
    from .common import file_sha
    upstream = library()
    components = {name: file_sha(inspect.getfile(getattr(upstream, name))) for name in
                  ('ActorPPOTrainer', 'CriticPPOTrainer', 'RemoteExperienceMaker',
                   'PolicyLoss', 'ValueLoss', 'compute_reward')}
    return {'name': 'openrlhf', 'version': VERSION, 'release_commit': REVISION,
            'integration': 'single_device_shared_actor_critic', 'source_sha256': components}


class ModelView(nn.Module):
    """Present the existing shared LoRA actor/value head using OpenRLHF's API."""
    def __init__(self, policy, critic=False):
        super().__init__()
        self.policy = policy
        self.critic = critic
        self.last_logprobs = None
        self.old_logprobs = None

    def train(self, mode=True):
        # The behavior distribution uses eval mode. LoRA/attention dropout are
        # disabled; retaining eval also prevents hidden base-model dropout.
        super().train(False)
        return self

    def forward(self, sequences, action_mask, attention_mask=None, **kwargs):
        logprobs, values = self.policy.statistics(
            sequences, attention_mask, sequences.shape[1] - action_mask.shape[1],
            with_values=self.critic)
        if not self.critic:
            ratio = (logprobs - self.old_logprobs)[action_mask]
            if not torch.isfinite(ratio).all() or ratio.abs().max() > 20:
                raise FloatingPointError("Nonfinite or extreme PPO likelihood ratio.")
            self.last_logprobs = logprobs.detach()
        return (values if self.critic else logprobs), SimpleNamespace()


class SingleDeviceStrategy:
    """Accumulate both upstream losses before stepping the shared parameters.

    OpenRLHF calls optimizer_step after each actor/critic backward. These calls
    register completion; finish() commits once after both losses and all micro-
    batches. Weighting by valid tokens preserves the full-minibatch objective.
    """
    ring_attn_group = None

    def __init__(self, policy, optimizer, config):
        self.policy, self.optimizer, self.config = policy, optimizer, config
        self.args = SimpleNamespace(use_dynamic_batch=False, aux_loss_coef=0.,
                                    entropy_loss_coef=None, use_kl_loss=False,
                                    advantage_estimator="gae", remote_rm_url=None,
                                    overlong_buffer_len=None, n_samples_per_prompt=1,
                                    reward_clip_range=None, gamma=config['gamma'],
                                    lambd=config['gae_lambda'], no_advantage_std_norm=False)
        # GAE needs no prompt grouping; 1 permits arbitrary counts after ungraded
        # responses are excluded. Generation still uses the configured repeats.
        self.scale = 1.
        self.calls = []

    def backward(self, loss, model, optimizer):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite OpenRLHF PPO loss.")
        weight = self.config['value_coefficient'] if model.critic else 1.
        (loss * self.scale * weight).backward()

    def optimizer_step(self, optimizer, model, scheduler, name):
        if optimizer is not self.optimizer or name != ('critic' if model.critic else 'actor'):
            raise RuntimeError("Unexpected OpenRLHF optimizer contract.")
        self.calls.append(name)

    def begin(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.calls.clear()

    def finish(self, microbatches):
        if self.calls != ['actor', 'critic'] * microbatches:
            raise RuntimeError("OpenRLHF did not complete both training steps for every microbatch.")
        norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['max_grad_norm'])
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite OpenRLHF PPO gradient norm.")
        self.optimizer.step()


def build_workers(policy, optimizer, config):
    upstream = library()
    strategy = SingleDeviceStrategy(policy, optimizer, config)
    actor, critic = ModelView(policy), ModelView(policy, critic=True)

    # The distributed constructors create CUDA replay buffers and collectives.
    # Supply only the state needed by their unmodified training_step methods;
    # the project owns batching and checkpointing on one device.
    class ActorWorker(upstream.ActorPPOTrainer):
        def __init__(self):
            self.strategy, self.args = strategy, strategy.args
            self.actor, self.actor_optim = actor, optimizer
            self.actor_scheduler = SimpleNamespace(get_last_lr=lambda: [optimizer.param_groups[0]['lr']])
            self.actor_loss_fn = upstream.PolicyLoss(
                clip_eps_low=config['clip_range'], clip_eps_high=config['clip_range'],
                token_level_loss=True, policy_loss_type='ppo', dual_clip=None,
                enable_vllm_is_correction=False)
            self.aux_loss, self.ema_model = False, None

    class CriticWorker(upstream.CriticPPOTrainer):
        def __init__(self):
            self.strategy, self.args = strategy, strategy.args
            self.critic, self.critic_optim = critic, optimizer
            self.critic_scheduler = SimpleNamespace(get_last_lr=lambda: [optimizer.param_groups[1]['lr']])
            self.critic_loss_fn = upstream.ValueLoss(config['value_clip_range'])
            self.aux_loss = False

    maker = upstream.RemoteExperienceMaker(None, None, None, None,
                SimpleNamespace(value=config['kl_coefficient']), strategy=strategy)
    return SimpleNamespace(actor=ActorWorker(), critic=CriticWorker(), strategy=strategy,
                           maker=maker, library=upstream)
