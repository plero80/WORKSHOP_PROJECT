from __future__ import annotations
from .common import ARMS, DEFAULT_CONFIG, OUTPUT_ROOT

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from .numeric import extract_answer, VERSION as NUMERIC_VERSION

from .answers import verify_answer
from .assets import bind_experiment, check_runtime, resolve_assets
from .common import (ROOT, atomic_json, digest, load_config, read_json, read_jsonl,
                     run_lock, seed_all, status, write_jsonl)
from .data import prepare_data, rollout_questions
from .memory import (GapMemory, Normalization, corrected_reward, select_memory,
                     fit_ridge, save_features, load_features)
from .metrics import summarize_rows
from .grading import number, scores, paired, subset, unavailable
from .models import Policy, RewardScorer, ScoreCache
from .ppo import PPOTrainer, load_checkpoint, save_checkpoint
from .teacher_memory import (teacher_config, load_matched_memory, prepare_teacher_memory, evaluate_teacher30b, shared_base_memory)


def serializable_response(row):
    return {k: v for k, v in row.items() if k not in ("prompt_ids", "response_ids", "embedding")}


def ensure_generation(policy, rows, output, name, repeats=1, greedy=False):
    folder = Path(output) / "generations" / name
    folder.mkdir(parents=True, exist_ok=True)
    all_items = []
    chunk_size = max(1, policy.config["generation"]["batch_size"] // repeats)
    for start in range(0, len(rows), chunk_size):
        started = time.monotonic()
        chunk = rows[start:start + chunk_size]
        path = folder / f"{start:06d}.json"
        input_hash = digest({"rows": chunk, "repeats": repeats, "greedy": greedy})
        was_cached = path.exists()
        if was_cached:
            saved = read_json(path)
            if saved["input_hash"] != input_hash:
                raise ValueError(f"Stale generated-response cache: {path}")
            generated = saved["items"]
        else:
            # Each block has a deterministic seed independent of earlier cache hits.
            seed_all(policy.config["seed"] + int(digest([name, start])[:8], 16))
            generated = policy.sample(chunk, repeats, greedy)
            atomic_json(path, {"input_hash": input_hash, "items": generated})
        all_items.extend(generated)
        elapsed = time.monotonic() - started
        rate = sum(x["response_tokens"] for x in generated) / max(elapsed, 1e-9)
        detail = "cached" if was_cached else f"{elapsed:.1f}s, {rate:.1f} answer tokens/s"
        print(f"Generate {name}: {min(start + chunk_size, len(rows))}/{len(rows)} questions; "
              f"{len(generated)} responses, {detail}", flush=True)
    return all_items


def score_both(items, proxy, judge, stage):
    print(f"Scoring {stage}: {len(items)} responses with proxy", flush=True)
    p = proxy.score(items, stage)
    print(f"Scoring {stage}: {len(items)} responses with judge", flush=True)
    j = judge.score(items, stage)
    emb = np.stack([x["embedding"] for x in p]).astype(np.float32)
    return scores([x["score"] for x in p]), scores([x["score"] for x in j]), emb, p, j


def annotated(items, ps, js, emb, norm, memory, identity, p_outputs=None, j_outputs=None):
    ps, js = scores(ps), scores(js)
    if memory is not None:
        pred, similarity, neighbors = memory.predict(emb, [x["id"] for x in items], identity)
    else:
        pred, similarity, neighbors = [None] * len(items), [None] * len(items), [[] for _ in items]
    gaps = norm.gap(ps, js) if norm is not None else [None] * len(items)
    result = []
    for i, item in enumerate(items):
        row = {**serializable_response(item), **verify_answer(item["response"], item["reference"]),
               "proxy_score": number(ps[i]), "judge_score": number(js[i]),
               "proxy_z": number(norm.proxy_z(ps[i])) if norm else None,
               "judge_z": number(norm.judge_z(js[i])) if norm else None,
               "gap": number(gaps[i]), "predicted_gap": number(pred[i]),
               "nearest_similarity": number(similarity[i]), "neighbor_indices": neighbors[i],
               "grading_status": "scored" if paired(ps[i], js[i]) else "unscored"}
        if p_outputs is not None:
            row["proxy_judgement"] = p_outputs[i]["judge_output"]
            row["judge_judgement"] = j_outputs[i]["judge_output"]
            row["proxy_grading_recovery"] = p_outputs[i].get("grading_recovery")
            row["judge_grading_recovery"] = j_outputs[i].get("grading_recovery")
            row["proxy_grading_format_recovery"] = p_outputs[i].get("grading_format_recovery")
            row["judge_grading_format_recovery"] = j_outputs[i].get("grading_format_recovery")
            row["proxy_review_path"] = p_outputs[i].get("review_path")
            row["judge_review_path"] = j_outputs[i].get("review_path")
        numeric = extract_answer(item["response"], length_capped=item.get("length_capped", False),
                                 ended_with_eos=item.get("ended_with_eos", True))
        prediction = numeric["prediction"]
        row.update(numeric_match=prediction is not None and prediction == row["gold_answer"],
                   numeric_unresolved=prediction is None, numeric_prediction=prediction,
                   numeric_method=numeric["method"], numeric_reason=numeric["reason"],
                   numeric_protocol=NUMERIC_VERSION, diagnostic_teacher="judge",
                   diagnostic_memory_teacher="judge")
        result.append(row)
    return result


def prepare_memory(policy, proxy, judge, split, config, output):
    output = Path(output)
    folder = output / "prepared"
    folder.mkdir(exist_ok=True)
    complete = folder / "complete.json"
    if complete.exists():
        norm = Normalization.load(folder / "normalization.json")
        memory = GapMemory.load(folder / "memory_initial.npz")
        if memory.encoder_identity != proxy.identity:
            raise ValueError("Memory does not match the proxy model/protocol.")
        probe = read_json(folder / "encoder_probe.json")
        current = proxy._infer([probe["row"]], "encoder_parity", config["scoring"]["max_new_tokens"])[0]
        similarity = float(np.dot(current["embedding"], np.asarray(probe["embedding"], np.float32)))
        if similarity < 0.999:
            raise ValueError(f"Saved proxy embedding cannot be reproduced (cosine={similarity:.6f}). Check model revisions, precision and tokenization.")
        return norm, memory
    cohorts = {}
    for name in ("calibration", "memory", "selection"):
        status(output, f"prepare_{name}")
        items = ensure_generation(policy, split["cohorts"][name], output, name,
                                  config["dataset"]["responses_per_prompt"])
        ps, js, emb, p, j = score_both(items, proxy, judge, name)
        cohorts[name] = (items, ps, js, emb, p, j)
        if name == "selection":
            save_features(folder / "selection_features.npz", items, emb, proxy.identity)
        write_jsonl(folder / f"{name}_raw.jsonl", [{**serializable_response(item),
                         **verify_answer(item["response"], item["reference"]),
                         "proxy_score": number(ps[i]), "judge_score": number(js[i]),
                         "proxy_review_path": p[i].get("review_path"), "judge_review_path": j[i].get("review_path"),
                         "proxy_judgement": p[i]["judge_output"], "judge_judgement": j[i]["judge_output"]}
                         for i, item in enumerate(items)])
        if name == "calibration":
            valid = paired(ps, js)
            if not valid.all() and (valid.sum() < 2 or min(ps[valid].std(), js[valid].std()) < config["scoring"]["minimum_std"]):
                unavailable(output, "prepared", "insufficient varied calibration grades", valid_pairs=int(valid.sum()), total=len(ps))
                norm = None
            else:
                norm = Normalization.fit(ps[valid], js[valid], config["knn"]["gap_quantile"], config["scoring"]["minimum_std"])
                norm.save(folder / "normalization.json")
    if norm is None:
        return None, None
    valid_cohorts = {name: subset(c, paired(c[1], c[2])) for name, c in cohorts.items()}
    atomic_json(folder / "grading_coverage.json", {
        name: {"total": len(c[0]), "valid_pairs": len(valid_cohorts[name][0]),
               "excluded": len(c[0]) - len(valid_cohorts[name][0])} for name, c in cohorts.items()})
    m, sel = valid_cohorts["memory"], valid_cohorts["selection"]
    if len(m[0]) < max(config["knn"]["k_grid"]) or not sel[0]:
        unavailable(output, "prepared", "insufficient valid memory or selection grades",
                    memory_pairs=len(m[0]), selection_pairs=len(sel[0]))
        return norm, None
    memory, grid = select_memory(m[3], norm.gap(m[1], m[2]), [x["id"] for x in m[0]],
                    sel[3], norm.gap(sel[1], sel[2]), [x["id"] for x in sel[0]], config, proxy.identity)
    memory.save(folder / "memory_initial.npz")
    atomic_json(folder / "selection_grid.json", grid)
    all_sel = cohorts["selection"]
    rows = annotated(*all_sel[:4], norm, memory, proxy.identity, *all_sel[4:])
    write_jsonl(folder / "selection_scored.jsonl", rows)
    atomic_json(folder / "selection_metrics.json", {**summarize_rows(rows, norm.threshold),
                "note": "Selection diagnostics; the default suite fixes k=32 and temperature=0.05 for both memories. Non-singleton grids are tuning metrics, not held-out evidence."})
    atomic_json(folder / "encoder_probe.json", {"row": serializable_response(m[0][0]), "embedding": m[3][0].tolist()})
    atomic_json(complete, {"encoder_identity": proxy.identity, "n_memory": len(memory.gaps)})
    return norm, memory


def smoke_ppo(policy, config, output, initial_state):
    """One discarded real update, followed by exact initial-state restoration."""
    path = Path(output) / 'preflight_ppo.json'
    if path.exists():
        return
    status(output, 'discarded_ppo_smoke')
    seed_all(config['seed'] + 713)
    rows = [{'id': 'smoke_a', 'question': 'What is 7 plus 8?', 'reference': '#### 15'},
            {'id': 'smoke_b', 'question': 'What is 9 minus 3?', 'reference': '#### 6'}]
    trainer = PPOTrainer(policy, config)
    try:
        items = policy.sample(rows)
        rollout = trainer.prepare(items, [1., -1.])
        parity = float(((rollout['old_logprobs'] - rollout['reference_logprobs']) * rollout['mask']).abs().max())
        result = trainer.optimize(rollout, 0)
        after = policy.trainable_state()
        changed = any(not torch.equal(after[part][key], initial_state[part][key])
                      for part in after for key in after[part])
        with torch.no_grad():
            reference, _ = policy.statistics(rollout['ids'], rollout['attention'], rollout['prompt_width'],
                                             reference=True, with_values=False)
        drift = float(((reference-rollout['reference_logprobs']) * rollout['mask']).abs().max())
        if parity > 2e-3 or not changed or drift > 2e-3:
            raise RuntimeError(f'PPO smoke failed: parity={parity}, changed={changed}, reference_drift={drift}')
        atomic_json(path, {'passed': True, 'initial_reference_error': parity, 'frozen_reference_max_drift': drift,
                          'updates_discarded': True, 'optimizer_steps': result['optimizer_steps']})
    finally:
        policy.restore_trainable(initial_state)
        policy.eval()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


def evaluate(policy, rows, proxy, judge, norm, memory, output, arm, step, kind, cached_items=None):
    folder = Path(output) / "evaluations" / kind / arm / f"step_{step:06d}"
    metrics_path = folder / "metrics.json"
    if metrics_path.exists():
        return read_json(metrics_path)
    status(output, f"evaluate_{kind}", arm=arm, update=step, questions=len(rows))
    items = cached_items if cached_items is not None else ensure_generation(
        policy, rows, output, f"{kind}/{arm}/{step:06d}", greedy=True)
    ps, js, emb, p, j = score_both(items, proxy, judge, f"{kind}/{arm}/{step}")
    scored = annotated(items, ps, js, emb, norm, memory, proxy.identity, p, j)
    folder.mkdir(parents=True, exist_ok=True)
    write_jsonl(folder / "responses.jsonl", scored)
    save_features(folder / "features.npz", items, emb, proxy.identity)
    metrics = {**summarize_rows(scored, norm.threshold if norm else None), "arm": arm, "update": step,
               "cohort": kind, "decoding": "greedy", "memory_examples": len(memory.gaps) if memory else 0,
               "numeric_protocol": NUMERIC_VERSION, "diagnostic_teacher": "judge",
               "diagnostic_memory_teacher": "judge"}
    atomic_json(metrics_path, metrics)
    print(f"{kind} {arm} at {step}: accuracy={metrics['accuracy']:.3%}, mean gap={metrics['mean_gap']}, unscored={metrics['n_unscored']}", flush=True)
    return metrics


def complete_cached_monitor(rows, folder, config):
    """Return exact historical answers only when every original block exists."""
    items = []
    size = config["generation"]["batch_size"]
    for start in range(0, len(rows), size):
        chunk = rows[start:start + size]
        path = Path(folder) / f"{start:06d}.json"
        if not path.exists():
            return None
        saved = read_json(path)
        expected = digest({"rows": chunk, "repeats": 1, "greedy": True})
        if saved["input_hash"] != expected or len(saved["items"]) != len(chunk):
            raise ValueError(f"Historical monitor cache does not match its cohort: {path}")
        for requested, generated in zip(chunk, saved["items"]):
            if any(requested[key] != generated[key] for key in ("id", "question", "reference")):
                raise ValueError(f"Historical monitor answer belongs to another question: {path}")
        items.extend(saved["items"])
    return items


def recover_cached_monitors(policy, rows, proxy, judge, norm, initial_memory, output, arms, config, ridge_model=None):
    output = Path(output)
    for arm in ["base", *arms]:
        for folder in sorted((output / "generations" / "monitor" / arm).glob("*")):
            if not folder.is_dir() or not folder.name.isdigit():
                continue
            step = int(folder.name)
            destination = output / "evaluations" / "monitor" / arm / f"step_{step:06d}"
            if (destination / "metrics.json").exists():
                continue
            items = complete_cached_monitor(rows, folder, config)
            if items is None:
                print(f"Monitor {arm}/{step} has incomplete generation; retaining it without inventing historical answers.", flush=True)
                continue
            memory = ridge_model if arm == "ridge" else initial_memory
            if arm == "knn_refresh":
                candidates = sorted(p for p in (output / "arms" / arm / "memories").glob("step_*.npz")
                                    if int(p.stem.split("_")[-1]) <= step)
                if candidates:
                    memory = GapMemory.load(candidates[-1])
            print(f"Recover monitor {arm}/{step}: grading {len(items)} saved answers; no regeneration.", flush=True)
            evaluate(policy, rows, proxy, judge, norm, memory, output, arm, step, "monitor", cached_items=items)
            atomic_json(destination / "grading_recovery.json", {
                "reason": "complete historical generations existed but scoring was unfinished",
                "responses_regenerated": 0, "arm": arm, "update": step, "time": time.time()})


def reward_for_arm(items, arm, proxy, judge, norm, memory, config):
    stage = f"training/{arm}"
    checked = [verify_answer(x["response"], x["reference"]) for x in items]
    details = [{**serializable_response(item), **check} for item, check in zip(items, checked)]
    if arm == "oracle":
        # Optional ground-truth reference. This arm intentionally uses 0/1 rewards.
        rewards = np.array([float(x["correct"]) for x in checked])
    elif arm == "judge":
        scores = judge.score(items, stage)
        rewards = norm.judge_z([x["score"] for x in scores])
        for row, score in zip(details, scores):
            row.update(judge_score=score["score"], judge_judgement=score["judge_output"])
            row["judge_grading_recovery"] = score.get("grading_recovery")
            row["judge_grading_format_recovery"] = score.get("grading_format_recovery")
            row["judge_review_path"] = score.get("review_path")
    else:
        scores = proxy.score(items, stage)
        zp = norm.proxy_z([x["score"] for x in scores])
        for row, score in zip(details, scores):
            row.update(proxy_score=score["score"], proxy_judgement=score["judge_output"])
            row["proxy_grading_recovery"] = score.get("grading_recovery")
            row["proxy_grading_format_recovery"] = score.get("grading_format_recovery")
            row["proxy_review_path"] = score.get("review_path")
        rewards = zp
        if arm.startswith("knn") or arm == "ridge":
            pred, similarities, _ = memory.predict(np.stack([x["embedding"] for x in scores]),
                                    [x["id"] for x in items], proxy.identity)
            rewards = corrected_reward(zp, pred, config["knn"]["correction"])
            for row, gap, sim in zip(details, pred, similarities):
                row.update(predicted_gap=float(gap), nearest_similarity=number(sim))
    penalties = config.get("completion_reward", {})
    adjusted = []
    for row, reward in zip(details, rewards):
        format_cost = penalties.get("format_penalty", 0.0) * (not row["format_valid"])
        incomplete_cost = penalties.get("incomplete_penalty", 0.0) * (
            row.get("length_capped", False) or not row.get("ended_with_eos", True))
        row.update(task_reward=number(reward), format_penalty=float(format_cost),
                   incomplete_penalty=float(incomplete_cost),
                   optimization_reward=number(reward - format_cost - incomplete_cost),
                   used_for_ppo=bool(np.isfinite(reward)),
                   exclusion_reason=None if np.isfinite(reward) else "unscored_reward")
        adjusted.append(row["optimization_reward"])
    rewards = np.asarray(adjusted, dtype=float)
    return rewards, details


def refresh_memory(policy, memory, proxy, judge, norm, split, config, output, step):
    path = Path(output) / "arms" / "knn_refresh" / "memories" / f"step_{step:06d}.npz"
    if path.exists():
        return GapMemory.load(path)
    k = config["knn"]
    pool = split["cohorts"]["refresh"]
    # Refresh prompts are permanently disjoint from PPO, monitor, selection and final.
    if not pool or k["refresh_prompts"] > len(pool):
        raise ValueError("Insufficient reserved refresh questions.")
    index = step // k["refresh_every"] - 1
    rows = rollout_questions(pool, index, k["refresh_prompts"], config["seed"] + 823)
    items = ensure_generation(policy, rows, output, f"refresh/{step:06d}", k["refresh_responses"])
    ps, js, emb, p, j = score_both(items, proxy, judge, f"refresh/knn_refresh/{step}")
    # Fixed calibration; no renormalization or retuning k on refreshed data.
    valid = paired(ps, js)
    updated = memory.extend(emb[valid], norm.gap(ps[valid], js[valid]), [x["id"] for i, x in enumerate(items) if valid[i]])
    updated.save(path)
    write_jsonl(path.with_suffix(".jsonl"), [{**serializable_response(x), "proxy_score": number(ps[i]),
                                           "judge_score": number(js[i]), "gap": number(norm.gap(ps[i], js[i])),
                                           "used_for_memory": bool(valid[i])}
                                          for i, x in enumerate(items)])
    return updated


def train_arm(policy, arm, target, proxy, judge, norm, initial_memory, initial_state,
              split, config, output, fingerprint, evaluation_context=None):
    eval_norm, eval_memory = evaluation_context or (norm, initial_memory)
    folder = Path(output) / "arms" / arm
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "checkpoint.pt"
    policy.restore_trainable(initial_state)
    trainer = PPOTrainer(policy, config)
    optimizer = trainer.optimizer
    step, memory, last_refresh, successful_updates = 0, initial_memory, 0, 0
    if checkpoint.exists():
        saved = load_checkpoint(checkpoint, policy, optimizer, fingerprint, arm)
        step = saved["step"]
        successful_updates = saved["extra"].get("successful_updates", step)
        last_refresh = saved["extra"].get("last_refresh", 0)
        if last_refresh:
            memory = GapMemory.load(folder / "memories" / f"step_{last_refresh:06d}.npz")
    if step > target:
        raise ValueError(f"{arm} already reached update {step}; requested {target} would mislabel a later policy. Request at least {step}.")
    p = config["ppo"]
    while step < target:
        start_time = time.monotonic()
        status(output, "ppo", arm=arm, update=step + 1, target=target)
        questions = rollout_questions(split["cohorts"]["ppo"], step, p["prompts_per_update"], config["seed"])
        seed_all(config["seed"] + 1_000_003 + step)
        generated_at = time.monotonic()
        items = policy.sample(questions, p["responses_per_prompt"])
        scored_at = time.monotonic()
        rewards, details = reward_for_arm(items, arm, proxy, judge, norm, memory, config)
        stats_at = time.monotonic()
        valid = np.isfinite(rewards)
        valid_items = [item for i, item in enumerate(items) if valid[i]]
        if valid_items:
            stats = trainer.update(valid_items, rewards[valid], successful_updates)
            successful_updates += 1
        else:
            stats = {"optimizer_steps": 0, "skip_reason": "no_valid_rewards",
                     "rollout_stats_seconds": 0., "optimization_seconds": 0.}
        stats.update(graded_responses=int(valid.sum()), excluded_responses=int((~valid).sum()),
                     successful_updates=successful_updates, skipped_updates=step + 1 - successful_updates)
        if torch.cuda.is_available() and policy.device.type == "cuda":
            torch.cuda.synchronize(policy.device)
        finished_at = time.monotonic()
        stats.update(generation_seconds=scored_at-generated_at,
                     grading_seconds=stats_at-scored_at)
        step += 1
        if arm == "knn_refresh" and step % config["knn"]["refresh_every"] == 0:
            memory = refresh_memory(policy, memory, proxy, judge, norm, split, config, output, step)
            last_refresh = step
        stats.update(update=step, arm=arm, mean_reward=float(np.mean(rewards[valid])) if valid.any() else None,
                     rollout_accuracy=float(np.mean([x["correct"] for x in details])),
                     seconds=time.monotonic() - start_time, memory_examples=len(memory.gaps) if memory else 0)
        # Per-update files are overwritten if an interrupted update must be replayed.
        atomic_json(folder / "training" / f"step_{step:06d}.json", stats)
        write_jsonl(folder / "rollouts" / f"step_{step:06d}.jsonl", details)
        do_monitor = step % p["monitor_every"] == 0 or step == target
        if step % p["checkpoint_every"] == 0 or do_monitor:
            save_checkpoint(checkpoint, policy, optimizer, step, fingerprint, arm,
                            {"last_refresh": last_refresh, "successful_updates": successful_updates})
        if do_monitor:
            evaluate(policy, split["cohorts"]["monitor"], proxy, judge, eval_norm, memory if arm == "knn_refresh" else eval_memory, output, arm, step, "monitor")
        print(f"PPO {arm} attempt {step}/{target}: reward={stats['mean_reward']}, excluded={stats['excluded_responses']}, rollout accuracy={stats['rollout_accuracy']:.3%}, {stats['seconds']:.1f}s", flush=True)
        print(f"  generate={stats['generation_seconds']:.1f}s grade={stats['grading_seconds']:.1f}s "
              f"old/ref+GAE={stats['rollout_stats_seconds']:.1f}s optimize={stats['optimization_seconds']:.1f}s", flush=True)
    # Also finishes a monitor interrupted immediately after its checkpoint save.
    evaluate(policy, split["cohorts"]["monitor"], proxy, judge, eval_norm, memory if arm == "knn_refresh" else eval_memory, output, arm, target, "monitor")
    policy.lm.save_pretrained(folder / "adapter")
    policy.tokenizer.save_pretrained(folder / "adapter")
    atomic_json(folder / "completed.json", {"update": step, "last_refresh": last_refresh, "fingerprint": fingerprint,
                                          "successful_updates": successful_updates, "skipped_updates": step - successful_updates})
    del optimizer, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return memory


def main(argv=None):
    parser = argparse.ArgumentParser(description="Workshop kNN: single-seed PPO experiment")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "main")
    parser.add_argument("--stage", choices=["pilot", "full", "prepare", "report"], default="full")
    parser.add_argument("--updates", type=int, help="Explicit target update; resumes existing arms to this target")
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        from .report import make_report
        make_report(output)
        return
    config = load_config(args.config)
    if args.updates is not None and args.updates < 1:
        parser.error("--updates must be positive")
    target = args.updates or config["ppo"]["pilot_updates" if args.stage == "pilot" else "full_updates"]
    arms = args.arms or config["arms"]
    if not set(arms) <= set(config["arms"]):
        parser.error("Requested arms must be declared in the config.")
    if len(set(arms)) != len(arms):
        parser.error("Duplicate arms")
    with run_lock(output):
        cache = None
        try:
            existing = output / "config.json"
            if existing.exists() and read_json(existing) != config:
                raise ValueError("Configuration changed. Choose a new --output directory.")
            atomic_json(existing, config)
            runtime = check_runtime(config)
            status(output, "runtime", **runtime)
            resolved = resolve_assets(config, output)
            split = prepare_data(config, output, resolved)
            fingerprint = bind_experiment(config, output, resolved, split, runtime)
            status(output, "load_policy_and_rewards")
            seed_all(config["seed"])
            policy = Policy(config, resolved)
            initial_path = output / "initial_trainable.pt"
            if initial_path.exists():
                initial_state = torch.load(initial_path, map_location="cpu", weights_only=True)
                policy.restore_trainable(initial_state)
            else:
                initial_state = policy.trainable_state()
                temp = initial_path.with_suffix(".tmp")
                torch.save(initial_state, temp)
                temp.replace(initial_path)
            smoke_ppo(policy, config, output, initial_state)
            cache = ScoreCache(output)
            proxy = RewardScorer("proxy", config, resolved, cache)
            judge = RewardScorer("judge", config, resolved, cache)
            norm, initial_memory = prepare_memory(policy, proxy, judge, split, config, output)
            strong_context = None
            if "knn_static_30b" in arms and norm is not None and initial_memory is not None:
                strong_context = load_matched_memory(output, norm, initial_memory, config, resolved)
                if strong_context is None:
                    status(output, "load_30b_memory_teacher")
                    teacher = RewardScorer("judge30b", teacher_config(config), resolved, cache)
                    try:
                        strong_context = prepare_teacher_memory(proxy, teacher, norm, initial_memory,
                                                                config, output, resolved)
                    finally:
                        del teacher
                        gc.collect()
                        torch.cuda.empty_cache()
                if strong_context is not None:
                    initial_memory = shared_base_memory(output, initial_memory)
            ridge_model = None
            if initial_memory is not None:
                initial_memory.save(output / 'prepared/memory_matched.npz')
                # Recompute selection predictions on the exact memory used by both arms.
                selection = read_jsonl(output / 'prepared/selection_raw.jsonl')
                features = load_features(output / 'prepared/selection_features.npz', selection, proxy.identity)
                selected_rows = annotated(selection, [r['proxy_score'] for r in selection],
                                          [r['judge_score'] for r in selection], features, norm, initial_memory, proxy.identity)
                write_jsonl(output / 'prepared/selection_scored.jsonl', selected_rows)
                atomic_json(output / 'prepared/selection_metrics.json', {
                    **summarize_rows(selected_rows, norm.threshold),
                    'note': 'Tuning diagnostics on the exact matched memory used for training, before PPO/final evaluation.'})
                if 'ridge' in arms:
                    status(output, 'fit_ridge')
                    ridge_model = fit_ridge(initial_memory, norm, output, config)
            from .validation import validate_saved_run
            validate_saved_run(output, include_final=False, config=config)
            recover_cached_monitors(policy, split["cohorts"]["monitor"], proxy, judge, norm,
                                    initial_memory, output, arms, config, ridge_model)
            evaluate(policy, split["cohorts"]["monitor"], proxy, judge, norm, initial_memory, output, "base", 0, "monitor")
            if args.stage == "prepare":
                status(output, "prepared")
                return
            final_marker = output / "final_protocol.json"
            if final_marker.exists():
                previous = read_json(final_marker)
                if previous["updates"] != target or previous["arms"] != arms:
                    raise ValueError("The final test set was already opened at a different target/arm list. Keep that declared protocol; use a new experiment for exploratory extensions.")
            # Pilot evaluates monitor only. The official test cohort stays unopened.
            skipped_arms = {}
            for arm in arms:
                reward_norm, reward_memory = (strong_context or (None, None)) if arm == "knn_static_30b" else (norm, initial_memory)
                if arm == "ridge":
                    reward_memory = ridge_model
                if (arm != "oracle" and reward_norm is None) or ((arm.startswith("knn") or arm == "ridge") and reward_memory is None):
                    skipped_arms[arm] = "insufficient valid grades to prepare this arm"
                    print(f"Skipping {arm}: {skipped_arms[arm]}; continuing.", flush=True)
                    continue
                train_arm(policy, arm, target, proxy, judge, reward_norm, reward_memory, initial_state,
                          split, config, output, fingerprint, evaluation_context=(norm, ridge_model if arm == "ridge" else initial_memory))
            atomic_json(output / "skipped_arms.json", skipped_arms)
            if args.stage == "full":
                atomic_json(final_marker, {"updates": target, "arms": arms, "decoding": "greedy",
                                          "questions": len(split["cohorts"]["final"]), "fingerprint": fingerprint})
                policy.restore_trainable(initial_state)
                evaluate(policy, split["cohorts"]["final"], proxy, judge, norm, initial_memory, output, "base", 0, "final")
                for arm in arms:
                    if arm in skipped_arms:
                        continue
                    trainer = PPOTrainer(policy, config)
                    optimizer = trainer.optimizer
                    saved = load_checkpoint(output / "arms" / arm / "checkpoint.pt", policy, optimizer, fingerprint, arm)
                    last = saved["extra"].get("last_refresh", 0)
                    memory = GapMemory.load(output / "arms" / arm / "memories" / f"step_{last:06d}.npz") if last else initial_memory
                    if arm == "ridge":
                        memory = ridge_model
                    evaluate(policy, split["cohorts"]["final"], proxy, judge, norm, memory, output, arm, target, "final")
                    del optimizer, trainer
                if strong_context is not None and config["teacher30b"]["evaluate_final"]:
                    destinations = [output / "evaluations" / "final" / a /
                                    f"step_{0 if a == 'base' else target:06d}" / "teacher30b_metrics.json"
                                    for a in ["base", *arms]]
                    if not all(p.exists() for p in destinations):
                        status(output, "load_30b_final_evaluator")
                        teacher = RewardScorer("judge30b", teacher_config(config), resolved, cache)
                        try:
                            evaluate_teacher30b(proxy, teacher, *strong_context, output, [a for a in arms if a not in skipped_arms], target)
                        finally:
                            del teacher
                            gc.collect()
                            torch.cuda.empty_cache()
            from .report import make_report
            make_report(output, target=target, arms=arms)
            status(output, "complete", run_stage=args.stage, updates=target, arms=arms,
                   skipped_arms=skipped_arms,
                   report=str(output / "report.md"))
        except Exception as e:
            status(output, "failed", error_type=type(e).__name__, message=str(e))
            traceback.print_exc()
            raise
        finally:
            if cache is not None:
                cache.close()


if __name__ == "__main__":
    main()
