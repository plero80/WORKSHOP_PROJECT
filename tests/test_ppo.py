"""Real tiny-model optimization, masking, frozen reference, and exact resume."""
import math
import pytest
import torch

from conftest import items
from workshop import ppo
from workshop.models import Policy
from workshop.ppo import PPOTrainer, load_checkpoint, save_checkpoint


def test_masked_gae_matches_hand_computed_terminal_rewards(tiny_assets):
    config, resolved = tiny_assets
    config['ppo']['gae_lambda'] = .5
    trainer = PPOTrainer(Policy(config, resolved), config)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    values = torch.zeros(2, 3)
    experience = trainer.workers.library.Experience(index=[0, 1], values=values,
        action_mask=mask, kl=torch.zeros_like(values), rewards=torch.tensor([1., -1.]),
        info={'response_length': mask.sum(1).float()})
    trainer.workers.maker.compute_advantages_and_returns([experience])
    advantages, returns = experience.advantages * mask, experience.returns
    expected = torch.tensor([[.5, 1., 0.], [-.25, -.5, -1.]])
    torch.testing.assert_close(returns, expected)
    assert advantages[~mask].count_nonzero() == 0
    assert abs(advantages[mask].mean().item()) < 1e-6
    assert math.isclose(advantages[mask].var(unbiased=False).item(), 1., abs_tol=1e-6)


@pytest.mark.parametrize('gradient_checkpointing', [False, True])
def test_update_and_exact_checkpoint_resume(tiny_assets, tmp_path, monkeypatch, gradient_checkpointing):
    config, resolved = tiny_assets
    config['runtime']['gradient_checkpointing'] = gradient_checkpointing
    config['runtime']['ppo_microbatch_size'] = 2
    policy = Policy(config, resolved)
    before = policy.trainable_state()
    frozen = {n: x.detach().clone() for n, x in policy.lm.named_parameters() if not x.requires_grad}
    trainer = PPOTrainer(policy, config)
    calls = {'reference': 0, 'gae': 0}
    statistics = policy.statistics
    gae = trainer.workers.maker.compute_advantages_and_returns

    def counted_statistics(*args, **kwargs):
        calls['reference'] += bool(kwargs.get('reference'))
        return statistics(*args, **kwargs)

    def counted_gae(*args):
        calls['gae'] += 1
        return gae(*args)

    monkeypatch.setattr(policy, 'statistics', counted_statistics)
    monkeypatch.setattr(trainer.workers.maker, 'compute_advantages_and_returns', counted_gae)
    rollout = trainer.prepare(items(), [1., -1.])
    torch.testing.assert_close(rollout['old_logprobs'], rollout['reference_logprobs'])
    result = trainer.optimize(rollout, 0)
    assert result['optimizer_steps'] == 2
    assert calls == {'reference': 1, 'gae': 1}
    after = policy.trainable_state()
    assert any(not torch.equal(before['adapter'][k], after['adapter'][k]) for k in before['adapter'])
    assert not torch.equal(before['value']['weight'], after['value']['weight'])
    for name, parameter in policy.lm.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    with torch.no_grad():
        reference, _ = policy.token_stats(items()[0], reference=True)
    torch.testing.assert_close(reference, rollout['reference_logprobs'][0, :2], rtol=1e-6, atol=1e-6)

    path = tmp_path/'checkpoint.pt'
    save_checkpoint(path, policy, trainer.optimizer, 1, 'identity', 'ridge')
    trainer.update(items(), [-.5, .8], 1)
    continued = policy.trainable_state()
    restored = PPOTrainer(policy, config)
    state = load_checkpoint(path, policy, restored.optimizer, 'identity', 'ridge')
    assert state['step'] == 1
    restored.update(items(), [-.5, .8], 1)
    for part, values in policy.trainable_state().items():
        for key, value in values.items():
            torch.testing.assert_close(value, continued[part][key], atol=0, rtol=0)
    with pytest.raises(ValueError, match='another experiment'):
        load_checkpoint(path, policy, restored.optimizer, 'wrong', 'ridge')
    with path.open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        load_checkpoint(path, policy, restored.optimizer, 'identity', 'ridge')


def test_microbatch_accumulation_preserves_update(tiny_assets):
    config, resolved = tiny_assets
    policy = Policy(config, resolved)
    initial = policy.trainable_state()
    outcomes = []
    for microbatch in (1, 2):
        policy.restore_trainable(initial)
        config['runtime']['ppo_microbatch_size'] = microbatch
        PPOTrainer(policy, config).update(items(), [1., -1.], 0)
        outcomes.append(policy.trainable_state())
    for part in outcomes[0]:
        for key in outcomes[0][part]:
            torch.testing.assert_close(outcomes[0][part][key], outcomes[1][part][key], atol=2e-6, rtol=2e-4)
