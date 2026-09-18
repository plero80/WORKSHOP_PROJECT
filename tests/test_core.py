from workshop.common import DEFAULT_CONFIG, OUTPUT_ROOT
import copy

import numpy as np
import pytest
import torch

from workshop.answers import numeric_value, parse_rating, verify_answer
from workshop.common import ROOT, load_config, append_jsonl, read_jsonl
from workshop.data import partition_rows, rollout_questions
from workshop.memory import GapMemory, Normalization, corrected_reward
from workshop.metrics import auroc, paired_bootstrap


@pytest.mark.parametrize("answer", [r"\boxed{12}", r"\boxed{12.0}", r"\boxed{\frac{24}{2}}", r"\boxed{1.2e1}"])
def test_equivalent_numbers(answer):
    assert verify_answer(answer, "working #### 12")["correct"]


@pytest.mark.parametrize("answer", ["The last number is 12", r"\boxed{11}", r"\boxed{12}\boxed{99}",
                                         r"\boxed{12 or 13}", r"\boxed{1/0}", r"\boxed{12 + 0}",
                                         r"\boxed{1,2}", r"\boxed{1e9999999}", r"\boxed{__import__('os')}"])
def test_checker_rejects_ambiguous_or_wrong_answers(answer):
    assert not verify_answer(answer, "#### 12")["correct"]


def test_rating_parser():
    assert parse_rating("Judgement: Correct.\nCorrectness_score: 5") == 5
    assert parse_rating("Judgement: The candidate says 5, but is wrong.\nCorrectness_score: 1") == 1
    assert parse_rating("5") is None
    assert parse_rating("Correctness_score: 5\nCorrectness_score: 2") is None
    assert parse_rating("Correctness_score: 50") is None


def test_frozen_normalization_and_signed_algebra():
    p, j = np.array([1, 2, 3, 4, 5]), np.array([2, 3, 4, 4, 4])
    norm = Normalization.fit(p, j, .95, .01)
    gap = norm.gap(p, j)
    np.testing.assert_allclose(corrected_reward(norm.proxy_z(p), gap), norm.judge_z(j))
    np.testing.assert_allclose(p - norm.proxy_std * gap, norm.proxy_mean + norm.proxy_std * norm.judge_z(j))
    assert corrected_reward([2], [-1], "signed")[0] == 3
    assert corrected_reward([2], [-1], "positive_only")[0] == 2
    with pytest.raises(ValueError, match="Degenerate"):
        Normalization.fit([5, 5], [1, 2], .95, .05)


def test_exact_neighbors_weights_exclusion_and_identity(tmp_path):
    e = np.array([[1, 0], [.8, .6], [0, 1], [-1, 0]], np.float32)
    m = GapMemory(e, [99, 2, 4, 8], ["same", "a", "b", "c"], 2, 1, "encoder")
    pred, similarity, neighbors = m.predict([[1, 0]], ["same"], "encoder")
    weights = np.exp([.8, 0]) / np.exp([.8, 0]).sum()
    assert neighbors == [[1, 2]]
    assert pred[0] == pytest.approx(weights @ [2, 4])
    assert similarity[0] == pytest.approx(.8)
    with pytest.raises(ValueError, match="mismatch"):
        m.predict([[1, 0]], encoder_identity="other")
    m.save(tmp_path / "memory.npz")
    restored = GapMemory.load(tmp_path / "memory.npz")
    np.testing.assert_allclose(restored.predict([[1, 0]], ["same"])[0], pred)
    expanded = m.extend([[0, -1]], [-4], ["new"])
    assert len(m.gaps) == 4 and len(expanded.gaps) == 5


def test_prompt_groups_are_disjoint_and_insufficient_data_fails():
    train = [{"question": f"Question {i}", "answer": "#### 1"} for i in range(16)]
    test = [{"question": "Question 0", "answer": "#### 1"}, {"question": "Another question", "answer": "#### 2"}]
    sizes = {name: 2 for name in ("calibration", "memory", "selection", "monitor", "refresh", "ppo", "final")}
    split = partition_rows(train, test, sizes, 42)
    sets = [set(x["id"] for x in rows) for rows in split["cohorts"].values()]
    assert len(set.union(*sets)) == sum(len(s) for s in sets)
    assert split["audit"]["train_test_overlap_excluded_from_train"] == 1
    assert split == partition_rows(train, test, sizes, 42)
    sizes["ppo"] = 30
    with pytest.raises(ValueError, match="Need"):
        partition_rows(train, test, sizes, 42)


def test_prompt_schedule_cycles_without_dropping_boundary_items():
    rows = [{"id": str(i)} for i in range(5)]
    flat = [x["id"] for update in range(5) for x in rollout_questions(rows, update, 2, 42)]
    assert sorted(flat[:5]) == sorted(flat[5:]) == ["0", "1", "2", "3", "4"]






def test_auc_ties_single_class_and_paired_evaluation():
    assert auroc([0, 1], [1, 2]) == 1
    assert auroc([0, 1], [2, 1]) == 0
    assert auroc([0, 1], [1, 1]) == .5
    assert auroc([0, 0], [1, 2]) is None
    left = [{"id": str(i), "correct": True} for i in range(4)]
    right = [{"id": str(i), "correct": False} for i in range(4)]
    result = paired_bootstrap(left, right, 100)
    assert result["accuracy_difference"] == 1 and result["ci95"] == [1., 1.]
    with pytest.raises(ValueError):
        paired_bootstrap(left, right[:2])


def test_default_configuration():
    c = load_config(DEFAULT_CONFIG)
    assert c["seed"] == 42 and c["arms"] == ["proxy", "judge", "knn_static", "knn_static_30b", "ridge"]
    assert c["generation"]["max_new_tokens"] == 768


def test_interrupted_budget_log_tail_is_recoverable(tmp_path):
    path = tmp_path / "events.jsonl"
    append_jsonl(path, {"n": 1})
    with path.open("a") as f:
        f.write('{"n":')
    assert read_jsonl(path) == [{"n": 1}]
    append_jsonl(path, {"n": 2})
    assert read_jsonl(path) == [{"n": 1}, {"n": 2}]
