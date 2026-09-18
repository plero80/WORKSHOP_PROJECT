"""Execution optimizations must preserve grading, optimizer math, and resume."""
import copy

import numpy as np
import pytest
import torch

from conftest import items
from workshop.assets import check_runtime
from workshop.common import DEFAULT_CONFIG, load_config, validate_config
from workshop.models import Policy, RewardScorer, ScoreCache
from workshop.ppo import PPOTrainer, load_checkpoint, save_checkpoint


def test_grading_batch_transactions_preserve_cache_and_order(tmp_path, monkeypatch):
    scorer = RewardScorer.__new__(RewardScorer)
    scorer.config = load_config(DEFAULT_CONFIG)
    scorer.config['scoring']['batch_size'] = 2
    scorer.role, scorer.identity = 'proxy', 'test-encoder'
    scorer.cache = ScoreCache(tmp_path)
    statements = []
    scorer.cache.db.set_trace_callback(statements.append)
    rows = [{'id': str(i), 'question': str(i), 'reference': '#### 1', 'response': '1'}
            for i in range(5)]
    calls = []

    def infer(batch, *args, **kwargs):
        calls.append([row['id'] for row in batch])
        return [{'score': float(int(row['id'])+1), 'judge_output': row['id'],
                 'embedding': np.array([int(row['id']), 1.], np.float32)} for row in batch]

    monkeypatch.setattr(scorer, '_infer', infer)
    result = scorer.score(rows, 'memory')
    assert [r['score'] for r in result] == [1., 2., 3., 4., 5.]
    assert calls == [['0', '1'], ['2', '3'], ['4']]
    assert sum(s == 'COMMIT' for s in statements) == 3
    scorer.cache.close()
    scorer.cache = ScoreCache(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError('A completed grading batch must be durable.')

    monkeypatch.setattr(scorer, '_infer', forbidden)
    cached = scorer.score(rows, 'resumed')
    for actual, expected in zip(cached, result):
        assert actual['score'] == expected['score']
        assert actual['judge_output'] == expected['judge_output']
        np.testing.assert_array_equal(actual['embedding'], expected['embedding'])
    # Serialization failure cannot partially commit an otherwise valid batch.
    with pytest.raises(TypeError):
        scorer.cache.put_many([('new', result[0]), ('invalid', {'embedding': None, 'bad': object()})])
    assert scorer.cache.get('new') is None
    scorer.cache.close()


@pytest.mark.parametrize('invalid', ['yes', 1])
def test_optimizer_switch_requires_boolean(invalid):
    config = load_config(DEFAULT_CONFIG)
    config['runtime']['fused_optimizer'] = invalid
    with pytest.raises(ValueError, match='fused_optimizer'):
        validate_config(config)


def test_cpu_tests_use_standard_optimizer(tiny_assets):
    config, resolved = tiny_assets
    assert config['runtime']['fused_optimizer'] is True
    trainer = PPOTrainer(Policy(config, resolved), config)
    assert trainer.optimizer.defaults['fused'] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device required')
def test_runtime_checks_training_kernels():
    config = load_config(DEFAULT_CONFIG)
    runtime = check_runtime(config)
    assert runtime['dtype'] == 'bfloat16' and runtime['attention'] == 'sdpa'
    assert runtime['fused_optimizer'] is True
    assert runtime['compute_capability'] == list(torch.cuda.get_device_capability())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device required')
def test_fused_cuda_update_parity_and_checkpoint_resume(tiny_assets, tmp_path):
    config, resolved = tiny_assets
    config['runtime'].update(device='cuda:0', dtype='bfloat16', attention='sdpa',
                             ppo_microbatch_size=2)
    policy = Policy(config, resolved)
    initial = policy.trainable_state()
    outcomes = []
    for fused in (False, True):
        policy.restore_trainable(initial)
        current = copy.deepcopy(config)
        current['runtime']['fused_optimizer'] = fused
        trainer = PPOTrainer(policy, current)
        assert trainer.optimizer.defaults['fused'] is fused
        trainer.update(items(), [1., -.5], 0)
        trainer.update(items(), [-.2, .8], 1)
        outcomes.append(policy.trainable_state())
    for part in outcomes[0]:
        for key in outcomes[0][part]:
            torch.testing.assert_close(outcomes[0][part][key], outcomes[1][part][key],
                                       atol=2e-6, rtol=2e-4)
    path = tmp_path/'checkpoint.pt'
    save_checkpoint(path, policy, trainer.optimizer, 2, 'fused-runtime', 'ridge')
    trainer.update(items(), [.3, -.4], 2)
    uninterrupted = policy.trainable_state()
    resumed = PPOTrainer(policy, current)
    load_checkpoint(path, policy, resumed.optimizer, 'fused-runtime', 'ridge')
    resumed.update(items(), [.3, -.4], 2)
    for part, values in policy.trainable_state().items():
        for key, value in values.items():
            torch.testing.assert_close(value, uninterrupted[part][key], atol=0, rtol=0)
