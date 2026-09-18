"""Hand-computed lower-tail errors, coverage, and matching predictor/policy tables."""
import json

import numpy as np
import pytest

from workshop.metrics import optimistic_tail_bias, TAIL_BIAS_KEYS
from workshop.report import predictor_policy_comparison, saved_metrics
from workshop.common import atomic_json, write_jsonl
from workshop.validation import evaluate_rows


def test_lower_prediction_tail_signed_residual_matches_excess_reward():
    predictions = np.arange(100, dtype=float)
    gaps = 2*predictions+1
    result = optimistic_tail_bias(gaps, predictions, [str(i) for i in range(100)])
    # Linear quantiles are .99, 4.95 and 9.9; <= selects 1, 5 and 10 answers.
    for key, n, bias in zip(TAIL_BIAS_KEYS, (1, 5, 10), (1., 3., 5.5)):
        assert result[key] == bias
        assert result[key+'_n'] == result[key+'_selected'] == result[key+'_questions'] == n
        assert result[key+'_fraction'] == n/100
        assert result[key+'_unscored'] == 0
        proxy = np.full(n, 2.)
        np.testing.assert_allclose((proxy-predictions[:n])-(proxy-gaps[:n]), gaps[:n]-predictions[:n])
    negative = optimistic_tail_bias(predictions-2, predictions)
    assert all(negative[key] == -2. for key in TAIL_BIAS_KEYS)
    perfect = optimistic_tail_bias(predictions, predictions)
    assert all(perfect[key] == 0. for key in TAIL_BIAS_KEYS)


def test_all_ties_are_retained_and_question_count_is_not_answer_count():
    predicted = [-2., -2., -2., 0., 1., 2., 3., 4., 5., 6.]
    gaps = [-1., 0., None, 0., 1., 2., 3., 4., 5., 6.]
    ids = ['same', 'same', 'missing', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    result = optimistic_tail_bias(gaps, predicted, ids)
    for key in TAIL_BIAS_KEYS:
        assert result[key] == 1.5
        assert result[key+'_selected'] == 3 and result[key+'_n'] == 2
        assert result[key+'_questions'] == 1 and result[key+'_unscored'] == 1
        assert result[key+'_fraction'] == .3
    constant = optimistic_tail_bias([1., 2., 3.], [0., 0., 0.])
    assert all(constant[key] == 2. and constant[key+'_selected'] == 3 for key in TAIL_BIAS_KEYS)


def test_missing_actual_gaps_do_not_redefine_the_tail():
    predicted = [-4., -3., -2., 0., np.nan, np.inf]
    missing = optimistic_tail_bias([None, 1., 3., 2., 1., 2.], predicted)
    complete = optimistic_tail_bias([0., 1., 3., 2., 1., 2.], predicted)
    assert missing['optimistic_tail_prediction_count'] == 4
    for key in TAIL_BIAS_KEYS:
        assert missing[key] is None
        assert missing[key+'_cutoff'] == complete[key+'_cutoff']
        assert missing[key+'_n'] == 0 and missing[key+'_selected'] == missing[key+'_unscored'] == 1
    json.dumps(missing, allow_nan=False)


@pytest.mark.parametrize('gaps,predictions', [([], []), ([None, np.nan, np.inf], [None, np.inf, np.nan])])
def test_no_finite_predictions_is_unavailable_and_json_safe(gaps, predictions):
    result = optimistic_tail_bias(gaps, predictions)
    for key in TAIL_BIAS_KEYS:
        assert result[key] is result[key+'_cutoff'] is None
        assert result[key+'_selected'] == result[key+'_n'] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('gaps,predictions,ids', [([1], [1, 2], None), ([[1]], [[1]], None), ([1], [1], ['a', 'b'])])
def test_alignment_errors_are_rejected(gaps, predictions, ids):
    with pytest.raises(ValueError, match='align'):
        optimistic_tail_bias(gaps, predictions, ids)


def test_tail_bias_does_not_require_classification_cutoffs_and_reports_reload_saved_answers(tmp_path):
    rows = [{'id': str(i), 'gap': float(i+1), 'predicted_gap': 0.} for i in range(4)]
    locked = {'label_definition': None, 'prediction_cutoff': None}
    result = evaluate_rows(rows, locked)
    assert result['high_gap_auroc'] is None
    assert result['optimistic_tail_bias_01'] == 2.5
    atomic_json(tmp_path/'metrics.json', {'accuracy': .5})
    write_jsonl(tmp_path/'responses.jsonl', rows)
    original = (tmp_path/'metrics.json').read_bytes()
    restored = saved_metrics(tmp_path/'metrics.json', tmp_path/'responses.jsonl')
    assert restored['optimistic_tail_bias_10'] == 2.5 and restored['accuracy'] == .5
    assert (tmp_path/'metrics.json').read_bytes() == original


def test_combined_table_uses_identical_base_answers_and_each_own_ppo_accuracy():
    predictions = [dict(cohort=cohort, predictor=method, gap_mse=mse)
                   for cohort, method, mse in [('final/base/0', 'ridge', .2),
                      ('final/base/0', 'knn_static', .3), ('final/ridge/400', 'ridge', .05)]]
    policies = [dict(cohort='final', arm=arm, accuracy=acc, numeric_accuracy=acc, n=100)
                for arm, acc in [('base', .1), ('knn_static', .5), ('ridge', .4)]]
    result = predictor_policy_comparison(predictions, policies)
    assert len(result) == 2
    assert {r['prediction_answers_from'] for r in result} == {'final/base/0'}
    ridge = next(r for r in result if r['ppo_policy'] == 'ridge')
    assert ridge['gap_mse'] == .2 and ridge['ppo_accuracy'] == .4
