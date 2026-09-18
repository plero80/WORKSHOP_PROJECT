"""Verify that the installed OpenRLHF code actually trains every reward arm."""
import inspect

import pytest
import torch

from conftest import items
from workshop.common import ENGINE_ID, DEFAULT_CONFIG, atomic_json, load_config, validate_config
from workshop.models import Policy
from workshop.ppo import PPOTrainer, load_checkpoint, save_checkpoint
from workshop.openrlhf_backend import library, VERSION
from workshop.suite import run_suite


def test_inherited_training_steps_execute_upstream_losses(tiny_assets, monkeypatch):
    config, resolved = tiny_assets
    trainer = PPOTrainer(Policy(config, resolved), config)
    upstream = library()
    assert trainer.workers.actor.training_step.__func__ is upstream.ActorPPOTrainer.training_step
    assert trainer.workers.critic.training_step.__func__ is upstream.CriticPPOTrainer.training_step
    assert 'openrlhf' in inspect.getfile(trainer.workers.actor.training_step)
    calls = {'actor': 0, 'critic': 0, 'advantages': 0}
    for key, owner, name in [('actor', trainer.workers.actor.actor_loss_fn, 'forward'),
                             ('critic', trainer.workers.critic.critic_loss_fn, 'forward'),
                             ('advantages', trainer.workers.maker, 'compute_advantages_and_returns')]:
        original = getattr(owner, name)
        def record(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(owner, name, record)
    result = trainer.update(items(), [1., -.5], 0)
    assert result['engine'] == ENGINE_ID and VERSION in ENGINE_ID
    assert calls == {'actor': 2, 'critic': 2, 'advantages': 1}


@pytest.mark.parametrize('count', [1, 3])
def test_partial_graded_rollouts_are_not_dropped(tiny_assets, count):
    config, resolved = tiny_assets
    trainer = PPOTrainer(Policy(config, resolved), config)
    rows = (items() * 2)[:count]
    result = trainer.update(rows, [float(i % 2) for i in range(count)], 0)
    assert result['rollout_response_tokens'] == sum(len(row['response_ids']) for row in rows)
    assert result['optimizer_steps'] == config['ppo']['epochs'] * ((count+1)//2)


def test_old_backend_checkpoints_and_suites_cannot_resume(tiny_assets, tmp_path):
    config, resolved = tiny_assets
    policy = Policy(config, resolved)
    trainer = PPOTrainer(policy, config)
    path = tmp_path/'checkpoint.pt'
    save_checkpoint(path, policy, trainer.optimizer, 0, 'test', 'ridge')
    atomic_json(path.with_suffix('.sha256.json'), {'engine': 'workshop.ppo.v1', 'sha256': 'old'})
    with pytest.raises(ValueError, match='engine mismatch'):
        load_checkpoint(path, policy, trainer.optimizer, 'test', 'ridge')
    atomic_json(tmp_path/'suite_protocol.json', {'config': config, 'seeds': [42]})
    with pytest.raises(ValueError, match='another PPO backend'):
        run_suite(config, tmp_path, [42, 43], 'prepare')


def test_config_has_no_silent_custom_ppo_fallback():
    config = load_config(DEFAULT_CONFIG)
    config['ppo']['backend'] = 'custom'
    with pytest.raises(ValueError, match='openrlhf'):
        validate_config(config)
