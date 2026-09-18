"""Matched labels, independent tuning, exact features, and online reward use."""
import copy
import numpy as np
import pytest
from sklearn.linear_model import Ridge

from workshop.common import DEFAULT_CONFIG, load_config, read_json, write_jsonl
from workshop.memory import GapMemory, Normalization, RidgeGap, fit_ridge, load_features, save_features
from workshop.run import reward_for_arm


def fixture_data(tmp_path):
    config = load_config(DEFAULT_CONFIG)
    config['ridge']['alphas'] = [.01, 1., 100.]
    norm = Normalization(3., 1., 3., 1., 1.)
    x = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], dtype=np.float32)
    memory = GapMemory(x, [1., -1., -1., 1.], ['a', 'b', 'c', 'd'], 2, .1, 'encoder')
    rows = [{'id': q, 'question': q, 'reference': '#### 1', 'response': str(i),
             'proxy_score': p, 'judge_score': j}
            for i, (q, p, j) in enumerate([('val1', 4, 3), ('val1', 3, 4), ('val2', 2, 4), ('missing', None, 3)])]
    write_jsonl(tmp_path/'prepared/selection_raw.jsonl', rows)
    save_features(tmp_path/'prepared/selection_features.npz', rows, x, 'encoder')
    # Deliberately unreadable final data: tuning must not access it.
    final = tmp_path/'evaluations/final/ridge/step_000400/responses.jsonl'
    final.parent.mkdir(parents=True)
    final.write_text('not JSON; not tuning data')
    return config, norm, memory, rows, x


def test_ridge_uses_same_memory_and_separate_question_weighted_selection(tmp_path, monkeypatch):
    config, norm, memory, rows, x = fixture_data(tmp_path)
    predictor = fit_ridge(memory, norm, tmp_path, config)
    selected = read_json(tmp_path/'prepared/ridge/selection.json')
    y = norm.gap([r['proxy_score'] for r in rows[:3]], [r['judge_score'] for r in rows[:3]])
    candidates = []
    for alpha in config['ridge']['alphas']:
        independent = Ridge(alpha=alpha, solver='cholesky').fit(memory.embeddings.astype(float), memory.gaps)
        candidates.append((np.average((independent.predict(x[:3])-y)**2, weights=[.5, .5, 1.]), -alpha))
    assert selected['alpha'] == -min(candidates)[1]
    independent = Ridge(alpha=selected['alpha'], solver='cholesky').fit(memory.embeddings.astype(float), memory.gaps)
    np.testing.assert_allclose(predictor.predict(x)[0], independent.predict(x))
    assert selected['memory_answers'] == 4 and selected['validation_excluded'] == 1
    assert selected['new_judge_calls'] == 0 and not selected['test_used'] and not selected['refit_on_validation']
    monkeypatch.setattr(Ridge, 'fit', lambda *a, **kw: pytest.fail('Resume must load frozen ridge'))
    np.testing.assert_array_equal(fit_ridge(memory, norm, tmp_path, config).coef, predictor.coef)
    memory.gaps[0] += 1
    with pytest.raises(ValueError, match='inputs changed'):
        fit_ridge(memory, norm, tmp_path, config)


def test_rejects_answer_encoder_or_split_mismatch(tmp_path):
    config, norm, memory, rows, x = fixture_data(tmp_path)
    changed = copy.deepcopy(rows)
    changed[0]['response'] = 'a different answer to the same prompt'
    for data, encoder in ((changed, 'encoder'), (rows, 'other_encoder')):
        with pytest.raises(ValueError, match='encoder/answers'):
            load_features(tmp_path/'prepared/selection_features.npz', data, encoder)
    memory.group_ids = np.array(['val1', 'b', 'c', 'd'])
    with pytest.raises(ValueError, match='overlaps'):
        fit_ridge(memory, norm, tmp_path, config)


def test_ridge_ppo_reward_needs_only_proxy_and_excludes_missing_scores(tmp_path):
    config, norm, memory, rows, x = fixture_data(tmp_path)
    ridge = fit_ridge(memory, norm, tmp_path, config)
    class Proxy:
        identity = 'encoder'
        def score(self, items, stage):
            return [{'score': 4. if i == 0 else None, 'embedding': x[i], 'judge_output': 'fixture reply'} for i in range(len(items))]
    class Judge:
        def score(self, *args):
            pytest.fail('Ridge PPO must not query the judge')
    rewards, details = reward_for_arm(rows[:2], 'ridge', Proxy(), Judge(), norm, ridge, config)
    assert rewards[0] == pytest.approx(1.-ridge.predict(x[:1])[0][0])
    assert np.isnan(rewards[1])
    assert details[1]['optimization_reward'] is None
