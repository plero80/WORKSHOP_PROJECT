"""Missing grades must not abort scoring, learning, evaluation or reporting."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import items
from workshop.common import DEFAULT_CONFIG, atomic_json, digest, load_config, read_json, read_jsonl, write_jsonl
from workshop.grading import paired
from workshop.memory import GapMemory, Normalization
from workshop.models import Policy, RewardScorer, ScoreCache
from workshop import run
from workshop.report import make_report
from workshop.teacher_memory import prepare_teacher_memory, load_matched_memory, shared_base_memory, assert_matched


@pytest.mark.parametrize('role', ['proxy', 'judge', 'judge30b'])
def test_exhausted_retries_save_full_review_continue_batch_and_cache(tiny_assets, tmp_path, monkeypatch, role):
    config, resolved = tiny_assets
    cache = ScoreCache(tmp_path)
    scorer = RewardScorer(role, config, resolved, cache)
    calls = []
    candidates = items()
    def infer(rows, stage, budget, retry=False):
        calls.append(budget)
        return [dict(score=None if row['id'] == candidates[0]['id'] else 4.,
                     judge_output='I cannot provide a grade.' if row['id'] == candidates[0]['id'] else 'Correctness_score: 4',
                     input_tokens=100, output_tokens=10, grading_length_capped=False,
                     embedding=np.array([1., 0.], np.float32) if role == 'proxy' else None) for row in rows]
    monkeypatch.setattr(scorer, '_infer', infer)
    result = scorer.score(candidates, 'memory')
    assert [x['score'] for x in result] == [None, 4.]
    case = read_json(tmp_path / result[0]['review_path'])
    assert case['question'] == candidates[0]['question']
    assert case['reference'] == candidates[0]['reference']
    assert case['response'] == candidates[0]['response']
    assert case['score'] is None and len(case['attempts']) == 5
    assert calls == [160, 320, 640, 1280, 2560]
    cache.close()
    scorer.cache = ScoreCache(tmp_path)
    again = scorer.score(candidates, 'selection')
    assert [x['score'] for x in again] == [None, 4.]
    assert calls == [160, 320, 640, 1280, 2560]
    assert len(list((tmp_path / 'review/ungraded').glob('*.json'))) == 1
    scorer.cache.close()


class Scorer:
    identity = 'proxy'
    def __init__(self, role='proxy', missing=()):
        self.role, self.missing = role, set(missing)
    def score(self, rows, stage):
        return [dict(score=None if row['id'] in self.missing else row[self.role + '_score'],
                     embedding=np.array(row['embedding'], np.float32), judge_output='saved reply') for row in rows]


def cohorts():
    return {name: [dict(id=f'{name}{i}', question=f'{name} {i}+1?', reference='#### 2',
                       response=r'\boxed{2}' if i % 2 == 0 else 'I do not know.',
                       response_tokens=3, length_capped=False, ended_with_eos=True,
                       proxy_score=[1, 2, 4, 5][i], judge_score=[2, 1, 5, 4][i],
                       embedding=np.eye(4)[i].tolist()) for i in range(4)]
            for name in ('calibration', 'memory', 'selection', 'monitor', 'refresh')}


def config():
    c = load_config(DEFAULT_CONFIG)
    c['knn']['k_grid'] = [2]
    return c


@pytest.mark.parametrize('all_missing', [False, True])
def test_preparation_and_evaluation_keep_unknown_rows_without_fake_labels(tmp_path, monkeypatch, all_missing):
    c, data = config(), cohorts()
    missing = [r['id'] for rows in data.values() for r in rows] if all_missing else ['calibration0', 'memory1', 'selection2']
    proxy, judge = Scorer(missing=missing), Scorer('judge')
    monkeypatch.setattr(run, 'ensure_generation', lambda policy, rows, *a, **kw: rows)
    norm, memory = run.prepare_memory(None, proxy, judge, {'cohorts': data}, c, tmp_path)
    if all_missing:
        assert norm is memory is None
    else:
        assert len(memory.gaps) == 3
        assert 'memory1' not in memory.group_ids
        metrics = read_json(tmp_path / 'prepared/selection_metrics.json')
        assert metrics['n'] == 4 and metrics['n_pair_scored'] == 3
        assert metrics['accuracy'] == .5
    metrics = run.evaluate(None, data['monitor'], proxy, judge, norm, memory, tmp_path, 'base', 0, 'monitor')
    assert metrics['n'] == 4 and metrics['accuracy'] == .5
    if all_missing:
        assert metrics['mean_proxy_score'] is None
        assert metrics['gap_mse'] is None
        assert metrics['n_unscored'] == 4
    atomic_json(tmp_path / 'config.json', c)
    make_report(tmp_path, target=0, arms=[])
    assert (tmp_path / 'report.md').exists()
    assert 'NaN' not in (tmp_path / 'summary.json').read_text()


def test_empty_then_partial_ppo_batch_advances_without_training_on_missing_rewards(tiny_assets, tmp_path, monkeypatch):
    c, resolved = tiny_assets
    c['ppo'].update(checkpoint_every=1, monitor_every=2)
    actor = Policy(c, resolved)
    initial = actor.trainable_state()
    norm = Normalization(3, 1, 3, 1, 1)
    memory = GapMemory(np.eye(2), [0, 1], ['a', 'b'], 1, .1, 'proxy')
    samples = items()
    monkeypatch.setattr(actor, 'sample', lambda *a, **kw: samples)
    monkeypatch.setattr(run, 'evaluate', lambda *a, **kw: {})
    calls = []
    def rewards(*args):
        valid = len(calls) > 0
        calls.append(valid)
        return np.array([np.nan, 1. if valid else np.nan]), [dict(correct=True), dict(correct=False)]
    monkeypatch.setattr(run, 'reward_for_arm', rewards)
    original_update = run.PPOTrainer.update
    def update(trainer, rows, rewards, index):
        assert len(rows) == 1 and rows[0] == samples[1]
        assert index == 0  # First actual update still checks the initial reference.
        return original_update(trainer, rows, rewards, index)
    monkeypatch.setattr(run.PPOTrainer, 'update', update)
    split = {'cohorts': {'ppo': samples, 'monitor': samples}}
    run.train_arm(actor, 'proxy', 2, None, None, norm, memory, initial, split, c, tmp_path, 'test')
    folder = tmp_path / 'arms/proxy'
    first, second = [read_json(folder / 'training' / f'step_{n:06d}.json') for n in (1, 2)]
    assert first['optimizer_steps'] == 0 and first['mean_reward'] is None
    assert second['optimizer_steps'] > 0 and second['excluded_responses'] == 1
    completed = read_json(folder / 'completed.json')
    assert completed['successful_updates'] == 1 and completed['skipped_updates'] == 1
    checkpoint = torch.load(folder / 'checkpoint.pt', weights_only=True)
    assert checkpoint['step'] == 2 and checkpoint['extra']['successful_updates'] == 1


def test_reward_details_preserve_missing_and_do_not_convert_it_to_penalty():
    c, data = config(), cohorts()['monitor']
    c['completion_reward'] = {'format_penalty': .5, 'incomplete_penalty': .5}
    norm = Normalization(3, 1, 3, 1, 1)
    rewards, details = run.reward_for_arm(data, 'proxy', Scorer(missing=['monitor1']), None, norm, None, c)
    assert np.isnan(rewards[1]) and details[1]['optimization_reward'] is None
    assert details[1]['task_reward'] is None and details[1]['used_for_ppo'] is False
    assert np.isfinite(rewards[[0, 2, 3]]).all()


@pytest.mark.parametrize('all_missing', [False, True])
def test_teacher_missing_labels_preserve_matched_geometry_or_skip_unavailable(tmp_path, all_missing):
    c, data = config(), cohorts()
    base_norm = Normalization.fit([1, 2, 4, 5], [2, 1, 5, 4], .95, .05)
    base = GapMemory(np.eye(4), base_norm.gap([1, 2, 4, 5], [2, 1, 5, 4]),
                     [x['id'] for x in data['memory']], 2, .05, 'proxy')
    for name in ('calibration', 'memory', 'selection'):
        write_jsonl(tmp_path / 'prepared' / f'{name}_raw.jsonl', data[name])
    missing = [r['id'] for rows in data.values() for r in rows] if all_missing else ['calibration1', 'memory2', 'selection3']
    result = prepare_teacher_memory(Scorer(), Scorer('judge', missing), base_norm, base, c, tmp_path, {'judge30b': 'revision'})
    if all_missing:
        assert result is None
        assert (tmp_path / 'prepared_30b/unavailable.json').exists()
    else:
        norm, memory = result
        assert len(memory.gaps) == 3
        shared = shared_base_memory(tmp_path, base)
        assert_matched(base_norm, shared, norm, memory)
        loaded_norm, loaded = load_matched_memory(tmp_path, base_norm, base, c, {'judge30b': 'revision'})
        assert loaded_norm == norm and np.array_equal(loaded.embeddings, shared.embeddings)
        metrics = read_json(tmp_path / 'prepared_30b/selection_metrics.json')
        assert metrics['n'] == 4 and metrics['n_unscored'] == 1


def test_all_ungraded_refresh_keeps_existing_memory(tmp_path, monkeypatch):
    c, data = config(), cohorts()
    c['knn'].update(refresh_prompts=4, refresh_responses=1)
    norm = Normalization(3, 1, 3, 1, 1)
    memory = GapMemory(np.eye(4), [0, 1, -1, 0], ['m1', 'm2', 'm3', 'm4'], 2, .05, 'proxy')
    monkeypatch.setattr(run, 'ensure_generation', lambda *a, **kw: data['refresh'])
    result = run.refresh_memory(None, memory, Scorer(missing=[r['id'] for r in data['refresh']]), Scorer('judge'),
                                norm, {'cohorts': data}, c, tmp_path, c['knn']['refresh_every'])
    np.testing.assert_array_equal(result.gaps, memory.gaps)
    np.testing.assert_array_equal(result.embeddings, memory.embeddings)


def test_main_completes_and_reports_unavailable_arms_when_all_grades_missing(tmp_path, monkeypatch):
    c, data = config(), cohorts()
    data['final'] = data['monitor']
    monkeypatch.setattr(run, 'load_config', lambda path: c)
    monkeypatch.setattr(run, 'check_runtime', lambda config: {})
    monkeypatch.setattr(run, 'resolve_assets', lambda *args: {})
    monkeypatch.setattr(run, 'prepare_data', lambda *args: {'cohorts': data})
    monkeypatch.setattr(run, 'bind_experiment', lambda *args: 'test')
    monkeypatch.setattr(run, 'smoke_ppo', lambda *args: None)
    monkeypatch.setattr(run, 'Policy', lambda *args: SimpleNamespace(
        trainable_state=lambda: {'weights': torch.zeros(1)}, restore_trainable=lambda state: None))
    missing = [r['id'] for rows in data.values() for r in rows]
    monkeypatch.setattr(run, 'RewardScorer', lambda role, *args: Scorer(role, missing))
    monkeypatch.setattr(run, 'ensure_generation', lambda policy, rows, *a, **kw: rows)
    run.main(['--output', str(tmp_path), '--stage', 'full', '--updates', '1'])
    assert read_json(tmp_path / 'status.json')['stage'] == 'complete'
    summary = read_json(tmp_path / 'summary.json')
    assert set(summary['skipped_arms']) == set(c['arms'])
    assert len(summary['metrics']) == 2
    assert all(r['n'] == 4 and r['n_unscored'] == 4 for r in summary['metrics'])


def test_suite_can_report_explicitly_unavailable_arms(tmp_path):
    from workshop.suite import aggregate
    c = config()
    for seed in (42, 43):
        folder = tmp_path / f'seed_{seed}'
        atomic_json(folder / 'final_protocol.json', {'arms': c['arms'], 'updates': 1})
        atomic_json(folder / 'summary.json', {
            'metrics': [{'arm': 'base', 'cohort': 'final', 'accuracy': .5, 'numeric_accuracy': .5,
                         'numeric_unresolved_rate': .5, 'format_valid_rate': .5, 'length_cap_rate': 0}],
            'skipped_arms': {a: 'insufficient valid grades' for a in c['arms']}})
    aggregate(tmp_path, [42, 43], c['arms'])
    result = read_json(tmp_path / 'suite_summary.json')
    assert result['arms']['proxy']['accuracy']['mean'] is None
    assert result['arms']['proxy']['accuracy']['n_seeds'] == 0
    assert result['arms']['base']['accuracy']['n_seeds'] == 2
