import copy
import json

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)

from workshop.common import atomic_json, read_json, write_jsonl
from workshop.validation import (DEFAULTS, evaluate_rows, fit_thresholds, intervals,
                                         validate_saved_run)


def lock(threshold=2):
    return {"label_definition": {"threshold": threshold, "comparator": ">"},
            "prediction_cutoff": {"threshold": 1., "comparator": ">="}, "calibration_mean_gap": 0.}


def test_regression_classification_and_ties_match_independent_library():
    truth, pred = np.array([-1, 0, 2, 3, 5.]), np.array([0, 0, 1, 2, 1.])
    rows = [dict(id=str(i), gap=g, predicted_gap=p, judge_z=-g, proxy_z=0.)
            for i, (g, p) in enumerate(zip(truth, pred))]
    result = evaluate_rows(rows, lock())
    y = truth > 2
    assert result['gap_mse'] == pytest.approx(mean_squared_error(truth, pred))
    assert result['gap_mae'] == pytest.approx(mean_absolute_error(truth, pred))
    assert result['gap_r2'] == pytest.approx(r2_score(truth, pred))
    assert result['gap_spearman'] == pytest.approx(spearmanr(truth, pred).statistic)
    assert result['high_gap_auroc'] == pytest.approx(roc_auc_score(y, pred))
    assert result['high_gap_average_precision'] == pytest.approx(average_precision_score(y, pred))
    assert result['balanced_accuracy'] == pytest.approx(balanced_accuracy_score(y, pred >= 1))
    assert result['corrected_judge_mse'] == pytest.approx(result['gap_mse'])


def test_label_or_decision_cutoffs_do_not_change_regression_and_decision_does_not_change_auc():
    rows = [dict(id=str(i), gap=float(i), predicted_gap=5.-i) for i in range(6)]
    a = evaluate_rows(rows, lock(2))
    changed = lock(2)
    changed['prediction_cutoff']['threshold'] = 4
    b = evaluate_rows(rows, changed)
    c = evaluate_rows(rows, lock(3))
    assert a['high_gap_auroc'] == b['high_gap_auroc']
    assert a['flagged_fraction'] != b['flagged_fraction']
    assert a['gap_mse'] == b['gap_mse'] == c['gap_mse']
    assert a['gap_r2'] == b['gap_r2'] == c['gap_r2'] < 0


def test_missing_nonfinite_constant_and_single_class_metrics_are_unavailable():
    rows = [dict(id='a', gap=1., predicted_gap=1.), dict(id='b', gap=1., predicted_gap=2.),
            dict(id='c', gap=None, predicted_gap=2.), dict(id='d', gap=float('nan'), predicted_gap=0.)]
    result = evaluate_rows(rows, lock())
    assert result['n'] == 4 and result['n_scored'] == result['n_unscored'] == 2
    assert result['gap_r2'] is result['gap_pearson'] is result['high_gap_auroc'] is None
    assert result['gap_mse'] == .5
    assert evaluate_rows([], lock())['gap_mse'] is None
    assert fit_thresholds(np.array([]), rows, DEFAULTS)['status'] == 'unavailable'
    json.dumps(result, allow_nan=False)


def test_saturated_quantile_recovers_supported_upper_tail_and_counts_questions():
    cal = np.array([-2.] * 40 + [0.] * 40 + [3.] * 10 + [4.] * 10)
    rows = [dict(id=f'q{i}', gap=float(g), predicted_gap=float(g)) for i, g in enumerate(cal)]
    settings = DEFAULTS | {'minimum_class_examples': 5, 'minimum_class_questions': 5}
    result = fit_thresholds(cal, rows, settings)
    assert result['status'] == 'selected'
    original = next(r for r in result['label_candidates'] if r['quantile'] == .95 and r['comparator'] == '>')
    inclusive = next(r for r in result['label_candidates'] if r['quantile'] == .95 and r['comparator'] == '>=')
    assert original['auroc'] is None and inclusive['auroc'] == 1
    for row in rows:
        row['id'] = 'only_one_question'
    assert fit_thresholds(cal, rows, settings)['status'] == 'unavailable'


def test_bootstrap_resamples_whole_questions():
    # Repeating each answer in its question changes no question-bootstrap metric.
    rows = [dict(id=str(i), gap=float(i), predicted_gap=float(i) / 2) for i in range(8)]
    a = intervals(rows, lock(), 40, 42)
    b = intervals([r for r in rows for _ in range(2)], lock(), 40, 42)
    for metric in a:
        np.testing.assert_allclose(a[metric]['ci95'], b[metric]['ci95'])


def saved_run(root):
    config = {'seed': 42, 'evaluation': {'bootstrap_samples': 20},
              'validation': DEFAULTS | {'minimum_class_examples': 1, 'minimum_class_questions': 1}}
    atomic_json(root / 'config.json', config)
    atomic_json(root / 'prepared/normalization.json', dict(proxy_mean=0., proxy_std=1., judge_mean=0., judge_std=1., threshold=4.))
    for cohort in ['calibration', 'memory', 'selection']:
        rows = [dict(id=f'{cohort}{i}', proxy_score=float(i+1), judge_score=1.,
                     gap=float(i), predicted_gap=float(i)*.5, proxy_z=float(i+1), judge_z=1.) for i in range(5)]
        write_jsonl(root / f'prepared/{cohort}_raw.jsonl', rows)
        if cohort == 'selection':
            write_jsonl(root / 'prepared/selection_scored.jsonl', rows)
    final = [dict(id=f'final{i}', gap=float(i), predicted_gap=float(i)*.3,
                  proxy_score=float(i+1), judge_score=1., proxy_z=float(i+1), judge_z=1.) for i in range(5)]
    path = root / 'evaluations/final/base/step_000000/responses.jsonl'
    write_jsonl(path, final)
    return path, final


def test_saved_validation_freezes_selection_before_final_and_preserves_original_files(tmp_path):
    final_path, final = saved_run(tmp_path)
    original = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    dest = tmp_path / 'validation/test'
    first = validate_saved_run(tmp_path, dest)
    assert all(p.read_bytes() == content for p, content in original.items())
    locked = (dest / 'judge_selection.json').read_bytes()
    for r in final:
        r['predicted_gap'] = -99.
    write_jsonl(final_path, final)
    second = validate_saved_run(tmp_path, dest)
    assert (dest / 'judge_selection.json').read_bytes() == locked
    assert first['final'][0]['metrics']['gap_mse'] != second['final'][0]['metrics']['gap_mse']
    assert first['teachers']['judge']['posthoc'] is True
    assert first['teachers']['judge30b']['status'] == 'unavailable'
    rows = read_json(tmp_path / 'config.json')
    rows['validation']['decision_metric'] = 'f1'
    with pytest.raises(ValueError, match='changed'):
        validate_saved_run(tmp_path, dest, config=rows)


def test_overlap_rejected_and_unsupported_grades_do_not_fail_validation(tmp_path):
    final_path, final = saved_run(tmp_path)
    final[0]['id'] = 'selection0'
    write_jsonl(final_path, final)
    with pytest.raises(ValueError, match='overlap'):
        validate_saved_run(tmp_path)
    final[0]['id'] = 'final0'
    write_jsonl(final_path, final)
    selection_path = tmp_path / 'prepared/selection_scored.jsonl'
    write_jsonl(selection_path, [dict(id='ungraded', gap=None, predicted_gap=0.)])
    result = validate_saved_run(tmp_path, tmp_path / 'validation/missing')
    assert result['teachers']['judge']['status'] == 'unavailable'
    assert result['teachers']['judge']['metrics']['n_unscored'] == 1
    assert result['final'][0]['metrics']['gap_mse'] is not None
    assert result['final'][0]['metrics']['high_gap_auroc'] is None


def test_preparation_only_never_opens_final_answers(tmp_path):
    path, _ = saved_run(tmp_path)
    path.write_text('invalid final JSON that must not be opened', encoding='utf8')
    result = validate_saved_run(tmp_path, include_final=False)
    assert result['final'] == []


def test_reward_correctness_uses_continuous_corrected_reward_and_independent_labels():
    # The proxy ties all answers, but a small correction ranks correctness perfectly.
    # It need not predict the numeric gap accurately to break those ties.
    rows = [dict(id=str(i), correct=bool(i % 2), numeric_match=bool(i % 2),
                 proxy_z=1., judge_z=float(i % 2), gap=1.-i % 2,
                 predicted_gap=.2 if i % 2 == 0 else .1) for i in range(6)]
    a = evaluate_rows(rows, lock())
    b = evaluate_rows(rows, lock(-100))
    assert a['correctness_proxy_auroc'] == .5
    assert a['correctness_corrected_reward_auroc'] == 1.
    assert a['gap_r2'] < 0
    assert a['correctness_corrected_reward_auroc'] == b['correctness_corrected_reward_auroc']
    assert a['numeric_correctness_corrected_reward_auroc'] == 1.
    assert a['correctness_n'] == a['correctness_judge_n'] == 6


def test_reward_correctness_missing_grades_and_numeric_labels_are_separate():
    rows = [dict(id='a', correct=False, numeric_match=True, proxy_z=1., predicted_gap=.1, gap=None, judge_z=None),
            dict(id='b', correct=True, numeric_match=True, proxy_z=1., predicted_gap=.2, gap=0., judge_z=1.),
            dict(id='c', correct=None, numeric_match=None, proxy_z=1., predicted_gap=.3, gap=0.),
            dict(id='d', correct=False, proxy_z=None, predicted_gap=.4, gap=None)]
    metrics = evaluate_rows(rows, lock())
    assert metrics['correctness_n'] == metrics['correctness_excluded'] == 2
    assert metrics['correctness_judge_n'] == 1
    assert metrics['correctness_corrected_reward_auroc'] == 0
    assert metrics['correctness_judge_auroc'] is None
    assert metrics['numeric_correctness_corrected_reward_auroc'] is None
    json.dumps(metrics, allow_nan=False)
