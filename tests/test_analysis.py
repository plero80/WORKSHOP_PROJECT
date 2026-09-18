"""Independent arithmetic and cohort checks for the core paper analysis."""
from collections import defaultdict
import json

import numpy as np
import pytest

from workshop.analysis import (classification, distance_buckets, gap_statistics, high_reward_tail_bias,
    optimism_statistics, peak_to_final, question_bootstrap, ranking, top_tail, _reward_diagnostics)
from workshop.metrics import optimistic_tail_bias


def test_reward_tail_and_gap_tail_are_different_conditioning_events():
    pred = np.array([-5., -4., 0., 1.])
    residual = np.array([1., 2., 3., 4.])
    proxy = np.array([-100., -100., 0., 100.])
    rewards = proxy-pred
    top = high_reward_tail_bias(residual, rewards, ['a', 'b', 'c', 'd'])
    bottom = optimistic_tail_bias(pred+residual, pred)
    assert top['high_reward_tail_optimism_05'] == 4.
    assert bottom['optimistic_tail_bias_05'] == 1.
    assert top['high_reward_tail_optimism_20_selected'] == 1


def test_high_reward_ties_and_missing_grades_do_not_move_cutoff():
    a = high_reward_tail_bias([None, -1., 2.], [5., 5., 0.], ['a', 'a', 'c'])
    b = high_reward_tail_bias([4., -1., 2.], [5., 5., 0.], ['a', 'a', 'c'])
    key = 'high_reward_tail_optimism_01'
    assert a[key] == -1.
    assert a[key+'_cutoff'] == b[key+'_cutoff'] == 5.
    assert a[key+'_selected'] == 2 and a[key+'_unscored'] == 1
    assert a[key+'_questions'] == 1
    empty = high_reward_tail_bias([np.nan], [np.nan], ['a'])
    assert empty[key] is None and empty[key+'_n'] == 0
    json.dumps(empty, allow_nan=False)


def test_optimism_distribution_has_signed_errors_and_explicit_missing_support():
    result = optimism_statistics([-2., 0., 1., 2., np.nan])
    assert result['optimism_mean'] == .25
    assert result['optimism_median'] == .5
    assert result['optimism_p95'] == pytest.approx(1.85)
    assert result['optimism_fraction_gt_0'] == .5
    assert result['optimism_fraction_gt_0_5'] == .5
    assert result['optimism_fraction_gt_1'] == .25
    assert result['optimism_n'] == 4 and result['optimism_excluded'] == 1


def test_ranking_constant_series_and_finite_pairs():
    assert ranking([3, 2, 1, np.nan], [1, 2, 3, 0])['kendall'] == -1.
    assert ranking(np.full(100, .3), np.arange(100))['pearson'] is None
    constant = gap_statistics(np.full(100, .3), np.arange(100))
    assert constant['gap_r2'] is constant['gap_kendall'] is None
    assert classification([True, False], [1., 0.])['auroc'] == 1.
    assert classification([False, False], [1., 0.])['average_precision'] is None
    assert classification([True, False]*50, np.full(100, .3))['point_biserial'] is None


def test_reward_signal_comparisons_use_identical_answers_and_inclusive_ties():
    rows = [{'id': 'a', 'correct': True, 'numeric_match': True},
            {'id': 'b', 'correct': False, 'numeric_match': False},
            {'id': 'c', 'correct': False, 'numeric_match': True}]
    tables = defaultdict(list)
    _reward_diagnostics({'arm': 'ridge'}, rows,
        {'proxy': np.array([10., 9., 1.]), 'judge': np.array([np.nan, 1., 1.]), 'ridge': np.array([0., 1., 2.])},
        np.array([np.nan, 1., 1.]), tables)
    assert all(r['common_reward_answers'] == 2 for r in tables['reward_correctness_alignment'])
    judge = next(r for r in tables['top_reward_accuracy'] if r['reward_signal'] == 'judge' and r['top_fraction'] == .01)
    assert judge['selected_answers'] == 2 and judge['selected_fraction'] == 1.
    all_rows = [r for r in tables['top_reward_accuracy'] if r['top_fraction'] == 1. and r['correctness'] == 'correct']
    assert all(r['accuracy'] == 0. for r in all_rows)


def test_question_bootstrap_preserves_groups_and_handles_unsupported_tails():
    ids = ['a', 'b', 'c', 'd']
    values = np.array([0., 0., 1., 1.])
    a = question_bootstrap(ids, {'accuracy': values}, samples=100, seed=3)
    b = question_bootstrap(np.repeat(ids, 2), {'accuracy': np.repeat(values, 2)}, samples=100, seed=3)
    assert a == b
    tails = {'top': ('optimism', 'reward', .1, True)}
    unsupported = question_bootstrap(ids, {'optimism': [None]*4, 'reward': [1., 2., 3., 4.]}, samples=30, seed=1, tail_specs=tails)
    assert unsupported['top']['ci95'] is None and unsupported['top']['valid_resamples'] == 0
    # Fixed optimism must stay fixed despite selection and resampling.
    constant = question_bootstrap(ids, {'optimism': [2.]*4, 'reward': [1., 2., 3., 4.]}, samples=30, seed=1, tail_specs=tails)
    assert constant['top']['ci95'] == [2., 2.] and constant['top']['tail_cutoffs_recomputed']


def test_bootstrap_tail_cutoffs_are_recomputed_from_each_draw():
    ids = ['a', 'b', 'c']
    values = {'optimism': np.array([-2., 1., 5.]), 'reward': np.array([0., 10., 20.])}
    result = question_bootstrap(ids, values, samples=80, seed=9,
        tail_specs={'top': ('optimism', 'reward', .2, True)})['top']
    rng, expected = np.random.default_rng(9), []
    for ix in rng.integers(0, 3, size=(80, 3)):
        chosen, _ = top_tail(values['reward'][ix], .2)
        expected.append(values['optimism'][ix][chosen].mean())
    np.testing.assert_allclose(result['ci95'], np.quantile(expected, [.025, .975]))


def test_distance_ties_stay_together_instead_of_arbitrary_row_splitting():
    bins, _ = distance_buckets([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.])
    assert len(set(bins[:5])) == len(set(bins[5:])) == 1
    assert set(bins) == {0, 2}
    bins, _ = distance_buckets([np.nan, .1])
    assert bins.tolist() == [-1, 0]


def test_peak_to_final_does_not_compare_monitor_peak_with_test_endpoint():
    common = {'seed': 42, 'arm': 'ridge', 'estimator': 'ridge', 'cohort': 'monitor'}
    rows = [dict(common, step=25, strict_accuracy=.8, mean_judge=4., mean_optimized_reward=.5, optimism_mean=0.),
            dict(common, step=50, strict_accuracy=.6, mean_judge=3., mean_optimized_reward=1., optimism_mean=.5),
            dict(common, cohort='final', step=50, strict_accuracy=.99, mean_judge=5., mean_optimized_reward=2., optimism_mean=0.)]
    result = next(r for r in peak_to_final(rows) if r['metric'] == 'strict_accuracy')
    assert result['difference'] == pytest.approx(-.2)
    assert result['peak_step'] == 25 and result['last_monitor_step'] == 50
    assert result['mean_optimized_reward_change_since_peak'] == .5
    assert result['mean_judge_change_since_peak'] == -1.


def test_single_command_analyzes_multiple_saved_seeds_without_changing_sources(tmp_path, monkeypatch):
    import csv
    import shutil
    from workshop.__main__ import main
    from workshop.common import DEFAULT_CONFIG, atomic_json, file_sha, load_config, read_json, write_jsonl
    from workshop.memory import GapMemory, Normalization, fit_ridge, save_features
    from workshop import memory as memory_module
    from workshop.analysis import analyze
    root = tmp_path/'suite'
    source = root/'seed_42'
    c = load_config(DEFAULT_CONFIG)
    c['arms'] = ['ridge']
    c['ridge']['alphas'] = [.1, 1.]
    c['evaluation']['bootstrap_samples'] = 10
    atomic_json(source/'config.json', c)
    norm = Normalization(3., 1., 3., 1., .5)
    x = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], dtype=np.float32)
    bank = GapMemory(x, [1., -1., 0., 2.], ['m1', 'm2', 'm3', 'm4'], 2, .1, 'fixture')
    bank.save(source/'prepared/memory_matched.npz')
    norm.save(source/'prepared/normalization.json')
    def rows(kind, arm='base'):
        return [{'id': f'{kind}{i}', 'question': f'{kind} math {i}', 'reference': '#### 1',
            'response': f'{arm} response {i}', 'correct': i % 2 == 0, 'numeric_match': i % 2 == 0,
            'format_valid': True, 'response_tokens': i+3, 'ended_with_eos': True, 'length_capped': False,
            'proxy_score': float([4, 2, 3, 5][i]), 'judge_score': float([3, 2, 4, 1][i])} for i in range(4)]
    selection = rows('selection')
    for name in ('selection_raw', 'selection_scored'):
        write_jsonl(source/f'prepared/{name}.jsonl', selection)
    save_features(source/'prepared/selection_features.npz', selection, x, 'fixture')
    fit_ridge(bank, norm, source, c)
    atomic_json(source/'validation/gap_validation_v3/judge_selection.json', {'label_definition': {'threshold': .5, 'comparator': '>'}})
    for kind in ('monitor', 'final'):
        for arm in ('base', 'ridge'):
            for step in ([0] if arm == 'base' else ([25, 50] if kind == 'monitor' else [50])):
                folder = source/f'evaluations/{kind}/{arm}/step_{step:06d}'
                saved = rows(kind, arm)
                if arm == 'ridge':
                    saved[0]['judge_score'] = None
                write_jsonl(folder/'responses.jsonl', saved)
                save_features(folder/'features.npz', saved, x, 'fixture')
                atomic_json(folder/'metrics.json', {'arm': arm})
    shutil.copytree(source, root/'seed_43')
    atomic_json(root/'seed_43/config.json', c | {'seed': 43})
    atomic_json(root/'suite_protocol.json', {'seeds': [42, 43], 'config': c})
    original = {p: file_sha(p) for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(memory_module, 'fit_ridge', lambda *a, **k: pytest.fail('Analysis must use frozen ridge'))
    assert main(['analyze', '--output', str(root), '--bootstrap-samples', '10', '--no-plots']) == 0
    destination = root/'paper_results'
    assert all(file_sha(p) == checksum for p, checksum in original.items())
    audit = read_json(destination/'analysis.json')
    assert [r['seed'] for r in audit['seeds']] == [42, 43]
    assert audit['status'] == 'complete' and audit['new_judge_calls'] == 0
    with (destination/'tables/seed_summary.csv').open(encoding='utf-8') as stream:
        summary = list(csv.DictReader(stream))
    row = next(r for r in summary if r['table'] == 'gap_prediction' and r.get('arm') == 'ridge'
               and r.get('cohort') == 'final' and r.get('estimator') == 'ridge' and r['metric'] == 'optimism_mean')
    assert row['n_seeds'] == '2' and float(row['sample_sd']) == 0.
    previous = (destination/'tables/gap_prediction.csv').read_bytes()
    analyze(root, bootstrap_samples=10, figures=False)
    assert (destination/'tables/gap_prediction.csv').read_bytes() == previous
    with pytest.raises(ValueError, match='separate'):
        analyze(root, root)
