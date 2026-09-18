import copy
import numpy as np
import pytest
import torch
from conftest import items
from workshop.common import digest, read_jsonl
from workshop.models import Policy, RewardScorer, ScoreCache, generation_settings

def test_generation_distribution_matches_ppo_logprobs(tiny_assets):
    config, resolved = tiny_assets
    policy = Policy(config, resolved)
    policy.eval()
    pids = [2, 4, 5]
    settings = generation_settings(policy.tokenizer, policy.lm, 6, sample=True, temperature=config["generation"]["temperature"])
    with torch.no_grad():
        generated = policy.lm.generate(input_ids=torch.tensor([pids]), attention_mask=torch.ones(1, 3, dtype=torch.long),
                    generation_config=settings, return_dict_in_generate=True, output_scores=True)
        response = generated.sequences[0, len(pids):].tolist()
        item = {"prompt_ids": pids, "response_ids": response}
        lp, _ = policy.token_stats(item)
        behavioral = torch.stack([s[0].log_softmax(-1)[token] for s, token in zip(generated.scores, response)])
    torch.testing.assert_close(lp, behavioral, atol=2e-6, rtol=2e-6)



@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_batched_padding_logprobs_values_and_gradients_match_single(tiny_assets, attention):
    config, resolved = tiny_assets
    config["runtime"]["attention"] = attention
    policy = Policy(config, resolved)
    policy.train()
    # Nonzero adapters/value weights exercise the reference and both loss paths.
    with torch.no_grad():
        for name, param in policy.lm.named_parameters():
            if "lora_B" in name:
                param.normal_(0, .01)
        policy.value_head.weight.normal_(0, .01)
    rows = items() + [{"prompt_ids": [2], "response_ids": [4, 5, 6, 7, 3]}]
    for reference in (False, True):
        with torch.no_grad():
            batch = policy.token_stats_batch(rows, reference)
            singles = [policy.token_stats(x, reference) for x in rows]
        for got, want in zip(batch, singles):
            for a, b in zip(got, want):
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    gradients = []
    for batched in (False, True):
        policy.zero_grad(set_to_none=True)
        stats = policy.token_stats_batch(rows) if batched else [policy.token_stats(x) for x in rows]
        total = sum(len(x["response_ids"]) for x in rows)
        loss = sum((lp * torch.linspace(-1, 1, len(lp))).sum() + v.square().sum()
                   for lp, v in stats) / total
        loss.backward()
        gradients.append({n: p.grad.clone() for n, p in policy.named_parameters() if p.requires_grad and p.grad is not None})
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], atol=3e-6, rtol=3e-4)



def test_batched_generation_distribution_with_unequal_prompt_lengths(tiny_assets):
    config, resolved = tiny_assets
    config["runtime"]["attention"] = "sdpa"
    policy = Policy(config, resolved).eval()
    prompts = [[2, 4, 5, 7], [2, 6]]
    batch = policy.tokenizer.pad({"input_ids": prompts}, padding=True, return_tensors="pt")
    settings = generation_settings(policy.tokenizer, policy.lm, 6, sample=True,
                                   temperature=config["generation"]["temperature"])
    with torch.no_grad():
        out = policy.lm.generate(**batch, generation_config=settings,
                                return_dict_in_generate=True, output_scores=True)
        rows, expected = [], []
        for i, prompt in enumerate(prompts):
            suffix = out.sequences[i, batch["input_ids"].shape[1]:].tolist()
            end = next((j + 1 for j, token in enumerate(suffix) if token == 3), len(suffix))
            response = suffix[:end]
            rows.append({"prompt_ids": prompt, "response_ids": response})
            expected.append(torch.stack([score[i].log_softmax(-1)[token]
                                         for score, token in zip(out.scores, response)]))
        stats = policy.token_stats_batch(rows)
    for (lp, _), want in zip(stats, expected):
        torch.testing.assert_close(lp, want, atol=3e-6, rtol=3e-5)



def test_qwen_judges_embeddings_and_cache(tiny_assets, tmp_path):
    config, resolved = tiny_assets
    cache = ScoreCache(tmp_path)
    proxy = RewardScorer("proxy", config, resolved, cache)
    judge = RewardScorer("judge", config, resolved, cache)
    rows = items()
    ps = proxy.score(rows, "test")
    js = judge.score(rows, "test")
    assert all(1 <= x["score"] <= 5 for x in ps + js)
    assert all(x["embedding"] is None for x in js)
    assert np.stack([x["embedding"] for x in ps]).shape == (2, 32)
    np.testing.assert_allclose([np.linalg.norm(x["embedding"]) for x in ps], [1, 1], atol=1e-6)
    cached = proxy.score(rows, "test_cached")
    assert [x["score"] for x in cached] == [x["score"] for x in ps]
    np.testing.assert_allclose(cached[0]["embedding"], ps[0]["embedding"])
    with pytest.raises(ValueError, match="input tokens"):
        old = config["scoring"]["max_input_tokens"]
        config["scoring"]["max_input_tokens"] = 1
        try:
            proxy._infer(rows, "too_long", 2)
        finally:
            config["scoring"]["max_input_tokens"] = old
    cache.close()


def test_generated_grading_retry_never_uses_fake_reward(tiny_assets, tmp_path, monkeypatch):
    config, resolved = tiny_assets
    config["scoring"]["mode"] = "rationale_then_score"
    cache = ScoreCache(tmp_path)
    proxy = RewardScorer("proxy", config, resolved, cache)
    calls = []

    def bad(rows, stage, max_new_tokens, retry=False):
        calls.append(retry)
        return [{"score": None, "judge_output": "unparseable", "input_tokens": 10,
                 "output_tokens": 2, "embedding": np.ones(32) / np.sqrt(32)} for _ in rows]

    monkeypatch.setattr(proxy, "_infer", bad)
    result = proxy.score(items()[:1], "invalid_test")
    assert result[0]["score"] is None
    assert result[0]["grading_status"] == "unscored"
    assert calls == [False, True, True, True, True]
    from workshop.common import read_jsonl
    assert len(read_jsonl(tmp_path / "invalid_judge_outputs.jsonl")) == 5
    cache.close()


def test_truncated_grade_recovers_and_valid_cache_is_reused(tiny_assets, tmp_path, monkeypatch):
    from workshop.answers import parse_rating
    config, resolved = tiny_assets
    config["scoring"]["mode"] = "rationale_then_score"
    cache = ScoreCache(tmp_path)
    proxy = RewardScorer("proxy", config, resolved, cache)
    calls = []
    def infer(rows, stage, max_new_tokens, retry=False):
        calls.append(max_new_tokens)
        text = "Judgement: Arithmetic is wrong.\nCorrectness_score: "
        if max_new_tokens >= 640:
            text += "2"
        return [{"score": parse_rating(text), "judge_output": text,
                 "input_tokens": 100, "output_tokens": max_new_tokens if max_new_tokens < 640 else 330,
                 "grading_length_capped": max_new_tokens < 640,
                 "embedding": np.ones(32, np.float32)/np.sqrt(32)} for _ in rows]
    monkeypatch.setattr(proxy, "_infer", infer)
    result = proxy.score(items()[:1], "monitor/judge/325")
    assert calls == [160, 320, 640]
    assert result[0]["score"] == 2
    assert result[0]["grading_recovery"]["max_new_tokens"] == 640
    again = proxy.score(items()[:1], "cached")
    assert calls == [160, 320, 640]
    assert again[0]["score"] == 2
    cache.close()


@pytest.mark.parametrize("success_budget", [160, 320])
def test_original_successful_attempts_never_use_extended_retries(tiny_assets, tmp_path, monkeypatch, success_budget):
    config, resolved = tiny_assets
    config["scoring"]["mode"] = "rationale_then_score"
    cache = ScoreCache(tmp_path)
    judge = RewardScorer("judge", config, resolved, cache)
    calls = []
    def infer(rows, stage, max_new_tokens, retry=False):
        calls.append(max_new_tokens)
        return [{"score": 4 if max_new_tokens >= success_budget else None,
                 "judge_output": "Correctness_score: 4" if max_new_tokens >= success_budget else "Judgement:",
                 "input_tokens": 10, "output_tokens": 10, "embedding": None} for _ in rows]
    monkeypatch.setattr(judge, "_infer", infer)
    results = judge.score(items()[:1], "training/judge")
    assert calls == ([160] if success_budget == 160 else [160, 320])
    assert results[0]["score"] == 4 and "grading_recovery" not in results[0]
    cache.close()





def test_pilot_then_full_runner_with_cpu_qwen_fixtures(tiny_assets, tmp_path, monkeypatch):
    from workshop import run
    from workshop.common import atomic_json, digest, read_json
    from workshop.data import partition_rows
    config, resolved = tiny_assets
    resolved["dataset"] = "offline_fixture"
    config["dataset"].update(calibration=3, memory=4, selection=3, monitor=2, refresh=2, ppo=4, final=2,
                             responses_per_prompt=2)
    config["scoring"]["minimum_std"] = 1e-9
    config["teacher30b"]["evaluate_final"] = True
    config["knn"].update(k_grid=[1, 2], temperature_grid=[.1], refresh_every=1, refresh_prompts=2, refresh_responses=1)
    config["ppo"].update(prompts_per_update=1, responses_per_prompt=2, checkpoint_every=1, monitor_every=1)
    config["arms"] = ["proxy", "judge", "knn_static", "knn_static_30b", "ridge", "knn_refresh", "oracle"]
    config["evaluation"]["bootstrap_samples"] = 20
    configpath = tmp_path / "config.json"
    atomic_json(configpath, config)
    train = [{"question": "math " + " ".join([str(i % 10)] * (i + 1)), "answer": "#### 1"} for i in range(20)]
    test = [{"question": "test math 2", "answer": "#### 2"}, {"question": "test math 3", "answer": "#### 3"}]
    split = partition_rows(train, test, config["dataset"], 42)
    split["fingerprint"] = digest(split)
    monkeypatch.setattr(run, "check_runtime", lambda c: {"gpu": "OFFLINE TEST: random tiny CPU Qwen"})
    monkeypatch.setattr(run, "resolve_assets", lambda c, o: resolved)
    monkeypatch.setattr(run, "prepare_data", lambda c, o, r: split)
    # Fixed scalar labels isolate runner mechanics from random-model grading skill.
    original_score = RewardScorer.score

    def varied_scores(self, rows, stage):
        out = original_score(self, rows, stage)
        for row, value in zip(rows, out):
            n = int(digest([row["id"], row["response"]])[:8], 16)
            value["score"] = float(1 + (n % 5 if self.role == "proxy" else (n // (25 if self.role == "judge30b" else 5)) % 5))
        return out

    monkeypatch.setattr(RewardScorer, "score", varied_scores)
    output = tmp_path / "experiment"
    common = ["--config", str(configpath), "--output", str(output)]
    run.main(common + ["--stage", "pilot", "--updates", "1"])
    assert not (output / "final_protocol.json").exists()
    assert not (output / "evaluations" / "final").exists()
    assert read_json(output / "status.json")["stage"] == "complete"
    run.main(common + ["--stage", "full", "--updates", "2"])
    for arm in config["arms"]:
        assert read_json(output / "arms" / arm / "completed.json")["update"] == 2
        assert (output / "evaluations" / "final" / arm / "step_000002" / "responses.jsonl").exists()
    assert read_json(output / "arms" / "knn_refresh" / "completed.json")["last_refresh"] == 2
    assert (output / "learning_curves.png").exists()
    report = read_json(output / "summary.json")
    assert len(report["metrics"]) == 16
    assert len(report["teacher30b_metrics"]) == 8
    assert "final/knn_static_30b_minus_knn_static/numeric" in report["paired_comparisons"]
    from workshop.memory import GapMemory
    m4 = GapMemory.load(output / "prepared" / "memory_initial.npz")
    m30 = GapMemory.load(output / "prepared_30b" / "memory_initial.npz")
    np.testing.assert_array_equal(m4.embeddings, m30.embeddings)
    np.testing.assert_array_equal(m4.group_ids, m30.group_ids)
    assert (m4.k, m4.temperature) == (m30.k, m30.temperature)
    ridge = read_json(output / "prepared/ridge/selection.json")
    assert ridge["memory_answers"] == len(m4.gaps)
    assert not ridge["test_used"] and not ridge["refit_on_validation"]
    assert "final/ridge_minus_knn_static/numeric" in report["paired_comparisons"]
    comparisons = read_json(output / "predictors/summary.json")["metrics"]
    assert {r["predictor"] for r in comparisons} == {"ridge", "knn_static"}
    assert sum(r["cohort"].startswith("final/") for r in comparisons) == 16
    assert all('optimistic_tail_bias_01' in r and 'optimistic_tail_bias_10_n' in r for r in comparisons)
    assert {r['predictor'] for r in report['predictor_policy_comparison']} == {'ridge', 'knn_static'}
    assert all(r['prediction_answers_from'] == 'final/base/0' for r in report['predictor_policy_comparison'])
    assert (output/'predictor_policy_comparison.csv').exists()
    import gzip
    import json
    from workshop.metrics import optimistic_tail_bias, TAIL_BIAS_KEYS
    with gzip.open(output/'predictors/all_predictions.jsonl.gz', 'rt', encoding='utf-8') as stream:
        exported = [json.loads(line) for line in stream]
    for metric in comparisons:
        matching = [r for r in exported if r['cohort'] == metric['cohort'] and r['predictor'] == metric['predictor']]
        recomputed = optimistic_tail_bias([r['gap'] for r in matching], [r['predicted_gap'] for r in matching], [r['id'] for r in matching])
        for key in TAIL_BIAS_KEYS:
            assert metric[key] == recomputed[key]
            assert sum(r[key+'_member'] for r in matching) == metric[key+'_selected']
    ridge_weights = (output / "prepared/ridge/ridge.npz").read_bytes()
    completed = output / "arms" / "knn_static_30b" / "checkpoint.pt"
    before_bytes = completed.read_bytes()
    run.main(common + ["--stage", "full", "--updates", "2"])
    assert completed.read_bytes() == before_bytes
    assert (output / "prepared/ridge/ridge.npz").read_bytes() == ridge_weights
    from workshop.common import read_jsonl
    events = read_jsonl(output / "judge_calls.jsonl")
    assert not any(e["role"] != "proxy" and e["stage"] in ("training/knn_static_30b", "training/ridge") for e in events)
    # The core analysis must read every saved monitor checkpoint without model calls.
    from workshop.analysis import analyze
    from workshop.common import file_sha
    import csv
    before = {str(p): file_sha(p) for p in output.rglob('*') if p.is_file()}
    monkeypatch.setattr(RewardScorer, 'score', lambda *a, **kw: pytest.fail('Offline analysis cannot call graders'))
    monkeypatch.setattr(Policy, 'sample', lambda *a, **kw: pytest.fail('Offline analysis cannot generate answers'))
    analyzed = analyze(output, bootstrap_samples=20)
    assert all(file_sha(p) == checksum for p, checksum in before.items())
    assert read_json(analyzed/'analysis.json')['status'] == 'complete'
    with (analyzed/'raw/ppo_checkpoint_metrics.csv').open(encoding='utf-8') as stream:
        trajectory = list(csv.DictReader(stream))
    ridge_monitors = [r for r in trajectory if r['cohort'] == 'monitor' and r['arm'] == r['estimator'] == 'ridge']
    assert {int(r['step']) for r in ridge_monitors} == {1, 2}
    assert {r['estimator'] for r in trajectory} == {'mean_gap', 'knn_static', 'ridge'}
    assert (analyzed/'figures/fig_top_reward_accuracy.pdf').exists()
    assert (analyzed/'stats/bootstrap_cis.json').exists()
    assert (analyzed/'tables/optimization_robustness.csv').exists()
    with pytest.raises(ValueError, match="final test set"):
        run.main(common + ["--stage", "pilot", "--updates", "3"])
