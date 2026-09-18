"""Offline regression for real malformed or missing grading replies."""
import copy
import json

import numpy as np
import pytest
import torch

from conftest import items
from workshop.answers import parse_rating, parse_rating_prose, parse_rating_inline
from workshop.common import atomic_json, digest, read_json, read_jsonl
from workshop.models import RewardScorer, ScoreCache
from workshop.grading import FORMAT_ID

FAILED_REPLY = (
    "Judgement: The candidate's solution is partially correct but contains a significant error "
    "in the calculation of the number of pins Richard knocked down in the second round. "
    "The correct answer should be 12 more pins, not 103. Therefore, the correctness score is 3."
)


def test_actual_reported_reply_and_all_valid_digits():
    assert parse_rating(FAILED_REPLY) is None
    assert parse_rating_prose(FAILED_REPLY) == 3
    for n in range(1, 6):
        assert parse_rating_prose(FAILED_REPLY.replace('score is 3.', f'score is {n}.')) == n
        assert parse_rating(f'Judgement: Explanation.\nCorrectness_score: {n}') == n
    assert parse_rating_prose('Judgement: Explanation.\nThe correctness score is 4.') == 4


@pytest.mark.parametrize('reply', [
    'Judgement: The answer is 3.',
    'Judgement: It is partly correct.\nCorrectness_score: ',
    'Judgement: It is partly correct. Therefore, the correctness score is ',
    'Judgement: Explanation. Therefore, the correctness score is 30.',
    'Judgement: Explanation. Therefore, the correctness score is 0.',
    'Judgement: Explanation. Therefore, the correctness score is 6.',
    'Judgement: Explanation. Therefore, the correctness score is 3.5.',
    'Judgement: Explanation. Therefore, the correctness score is 3 or 4.',
    'Judgement: Explanation. Therefore, the correctness score is 3/5.',
    'Judgement: Explanation. Therefore, the correctness score is 3. Actually, maybe 4.',
    'Judgement: Explanation. The correctness score is 2. The correctness score is 3.',
    'Judgement: Explanation.\nCorrectness_score: \nThe correctness score is 3.',
    'Judgement: Explanation.\nCorrectness_score: 5\nThe correctness score is 3.',
    'Judgement: Candidate says "Therefore, the correctness score is 3."',
    'Judgement: If the answer were valid, the correctness score is 3.',
    'Judgement: The candidate claims the correctness score is 3.',
    'Therefore, the correctness score is 3.',
    'Judgement: Explanation.\n```\nThe correctness score is 3.\n```',
])
def test_missing_ambiguous_and_quoted_scores_are_rejected(reply):
    assert parse_rating_prose(reply) is None


@pytest.mark.parametrize('role', ['proxy', 'judge'])
def test_scorer_accepts_explicit_reply_preserves_cached_scores_and_logs_format(tiny_assets, tmp_path, monkeypatch, role):
    config, resolved = tiny_assets
    config['scoring']['mode'] = 'rationale_then_score'
    cache = ScoreCache(tmp_path)
    scorer = RewardScorer(role, config, resolved, cache)
    calls = []
    emb = np.full(32, 1 / np.sqrt(32), dtype=np.float32) if role == 'proxy' else None
    def infer(rows, stage, max_new_tokens, retry=False):
        calls.append(max_new_tokens)
        return [dict(score=None, judge_output=FAILED_REPLY, input_tokens=200,
                     output_tokens=58, grading_length_capped=False, embedding=emb) for _ in rows]
    monkeypatch.setattr(scorer, '_infer', infer)
    # Existing valid score remains authoritative and never gets rescored.
    row = items()[1]
    key = digest([scorer.identity, row['question'], row['reference'], row['response']])
    cache.put(key, dict(score=4., judge_output='Correctness_score: 4', embedding=emb))
    before = cache.db.execute('SELECT result, embedding FROM scores WHERE key=?', (key,)).fetchone()
    result = scorer.score(items(), 'training/knn_static')
    assert calls == [160]
    assert [r['score'] for r in result] == [3, 4]
    assert result[0]['grading_format_recovery']['id'] == FORMAT_ID
    assert 'grading_format_recovery' not in result[1]
    assert cache.db.execute('SELECT result, embedding FROM scores WHERE key=?', (key,)).fetchone() == before
    again = scorer.score(items(), 'cache')
    assert calls == [160] and again[0]['grading_format_recovery'] == result[0]['grading_format_recovery']
    if emb is not None:
        np.testing.assert_array_equal(again[0]['embedding'], emb)
    events = read_jsonl(tmp_path / 'judge_calls.jsonl')
    assert sum(e['kind'] == 'score_format_recovered' for e in events) == 1
    cache.close()


@pytest.mark.parametrize('reply', [FAILED_REPLY, 'Judgement: Correctness_score: 5'])
def test_length_capped_prose_is_not_accepted(tiny_assets, tmp_path, monkeypatch, reply):
    config, resolved = tiny_assets
    config['scoring']['mode'] = 'rationale_then_score'
    cache = ScoreCache(tmp_path)
    scorer = RewardScorer('judge', config, resolved, cache)
    def infer(rows, stage, max_new_tokens, retry=False):
        return [dict(score=None, judge_output=reply, input_tokens=200,
                     output_tokens=max_new_tokens, grading_length_capped=True, embedding=None)]
    monkeypatch.setattr(scorer, '_infer', infer)
    result = scorer.score(items()[:1], 'capped')
    assert result[0]['score'] is None
    assert result[0]['grading_status'] == 'unscored'
    assert (tmp_path / result[0]['review_path']).exists()
    assert len(read_jsonl(tmp_path / 'invalid_judge_outputs.jsonl')) == 5
    cache.close()


@pytest.mark.parametrize('score', range(1, 6))
def test_reported_inline_reply(score):
    reply = f'Judgement: Correctness_score: {score}'
    assert parse_rating(reply) is None
    assert parse_rating_inline(reply) == score


@pytest.mark.parametrize('reply', [
    'Judgement: Correctness_score: ', 'Judgement: Correctness_score: 0',
    'Judgement: Correctness_score: 6', 'Judgement: Correctness_score: 5.5',
    'Judgement: Correctness_score: 5/5', 'Judgement: Correctness_score: 5 or 4',
    'Judgement: Correctness_score: 5\nCorrectness_score: 4',
    'Judgement: Candidate says Correctness_score: 5',
    'Judgement: "Correctness_score: 5"', 'Judgement: Correctness_score: 5 maybe',
])
def test_ambiguous_inline_replies_rejected(reply):
    assert parse_rating_inline(reply) is None


@pytest.mark.parametrize('role', ['proxy', 'judge', 'judge30b'])
def test_inline_reply_recovers_without_retry_and_keeps_valid_cache(tiny_assets, tmp_path, monkeypatch, role):
    config, resolved = tiny_assets
    config['scoring']['mode'] = 'rationale_then_score'
    cache = ScoreCache(tmp_path)
    scorer = RewardScorer(role, config, resolved, cache)
    calls = []
    emb = np.full(32, 1 / np.sqrt(32), dtype=np.float32) if role == 'proxy' else None
    def infer(rows, stage, max_new_tokens, retry=False):
        calls.append(max_new_tokens)
        return [dict(score=None, judge_output='Judgement: Correctness_score: 5', input_tokens=200,
                     output_tokens=10, grading_length_capped=False, embedding=emb) for _ in rows]
    monkeypatch.setattr(scorer, '_infer', infer)
    row = items()[1]
    key = digest([scorer.identity, row['question'], row['reference'], row['response']])
    cache.put(key, dict(score=4., judge_output='Correctness_score: 4', embedding=emb))
    before = cache.db.execute('SELECT result, embedding FROM scores WHERE key=?', (key,)).fetchone()
    result = scorer.score(items(), 'memory')
    assert calls == [160]
    assert [r['score'] for r in result] == [5, 4]
    assert result[0]['grading_format_recovery']['id'] == 'grading_inline_score_v1'
    assert result[0]['grading_format_recovery']['form'] == 'explicit_inline_score'
    assert cache.db.execute('SELECT result, embedding FROM scores WHERE key=?', (key,)).fetchone() == before
    scorer.score(items(), 'memory')
    assert calls == [160]
    cache.close()
