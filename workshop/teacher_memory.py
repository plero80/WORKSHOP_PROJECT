"""Matched static memories: the teacher labels change, the proxy geometry does not."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .common import atomic_json, digest, file_sha, read_json, read_jsonl, status, write_jsonl
from .memory import GapMemory, Normalization
from .models import RewardScorer
from .grading import number, scores, paired, unavailable


def teacher_config(config):
    result = copy.deepcopy(config)
    result["scoring"]["batch_size"] = config["teacher30b"]["batch_size"]
    return result


def same_examples(items):
    return digest([{key: x[key] for key in ("id", "question", "reference", "response")}
                   for x in items])


def load_matched_memory(output, base_norm, base_memory, config, resolved):
    folder = Path(output) / "prepared_30b"
    if not (folder / "complete.json").exists():
        return None
    manifest = read_json(folder / "complete.json")
    if manifest["teacher_model"] != config["models"]["judge30b"] or manifest["teacher_revision"] != resolved["judge30b"]:
        raise ValueError("30B memory teacher identity changed; use a new output directory.")
    for name in ("calibration", "memory", "selection"):
        source = read_jsonl(Path(output) / "prepared" / f"{name}_raw.jsonl")
        if same_examples(source) != manifest["shared_response_hashes"][name]:
            raise ValueError(f"30B memory no longer matches the 4B {name} responses.")
    for name, expected in manifest["artifact_sha256"].items():
        if file_sha(folder / name) != expected:
            raise ValueError(f"30B prepared artifact changed: {name}")
    norm = Normalization.load(folder / "normalization.json")
    memory = GapMemory.load(folder / "memory_initial.npz")
    assert_matched(base_norm, shared_base_memory(output, base_memory), norm, memory)
    return norm, memory


def shared_base_memory(output, base_memory):
    """Both memories use the same subset if a 30B memory label is unavailable."""
    path = Path(output) / "prepared_30b" / "complete.json"
    if not path.exists():
        return base_memory
    indices = read_json(path).get("shared_memory_indices")
    if indices is None:
        return base_memory  # compatible with earlier, fully graded preparations
    return GapMemory(base_memory.embeddings[indices], base_memory.gaps[indices],
                     base_memory.group_ids[indices], base_memory.k,
                     base_memory.temperature, base_memory.encoder_identity)


def assert_matched(base_norm, base_memory, norm, memory):
    if (norm.proxy_mean, norm.proxy_std) != (base_norm.proxy_mean, base_norm.proxy_std):
        raise ValueError("Both teacher memories must share the same proxy normalization.")
    if (memory.encoder_identity != base_memory.encoder_identity or
        memory.k != base_memory.k or memory.temperature != base_memory.temperature or
        not np.array_equal(memory.group_ids, base_memory.group_ids) or
        not np.array_equal(memory.embeddings, base_memory.embeddings)):
        raise ValueError("Teacher comparison must use identical proxy embeddings, IDs, k and temperature.")


def prepare_teacher_memory(proxy, teacher, base_norm, base_memory, config, output, resolved):
    """Label the already saved responses. Never sample extra candidates for this arm."""
    from .run import annotated
    from .metrics import summarize_rows

    folder = Path(output) / "prepared_30b"
    folder.mkdir(parents=True, exist_ok=True)
    cohorts, hashes = {}, {}
    for name in ("calibration", "memory", "selection"):
        status(output, f"prepare_30b_{name}")
        original = read_jsonl(Path(output) / "prepared" / f"{name}_raw.jsonl")
        if not original:
            raise ValueError(f"Missing original {name} responses; build the 4B memory first.")
        # Read proxy outputs from the same scorer/cache; its prompts and identity are unchanged.
        p = proxy.score(original, f"teacher30b/{name}/proxy_cache")
        ps = scores([x["score"] for x in p])
        if not np.array_equal(ps, scores([x["proxy_score"] for x in original]), equal_nan=True):
            raise ValueError("Proxy scores changed while relabeling the matched memory.")
        emb = np.stack([x["embedding"] for x in p]).astype(np.float32)
        base_valid = paired(ps, [x["judge_score"] for x in original])
        if name == "memory" and not np.array_equal(emb[base_valid], base_memory.embeddings):
            raise ValueError("Proxy embedding parity failed for the shared memory responses.")
        j = teacher.score(original, f"teacher30b/{name}")
        js = scores([x["score"] for x in j])
        cohorts[name] = (original, ps, js, emb, p, j)
        hashes[name] = same_examples(original)
        write_jsonl(folder / f"{name}_raw.jsonl", [
            {**item, "judge4b_score": item["judge_score"],
             "judge_score": number(js[i]), "judge_judgement": j[i]["judge_output"],
             "teacher_review_path": j[i].get("review_path"),
             "teacher_role": "judge30b", "teacher_grading_recovery": j[i].get("grading_recovery"),
             "teacher_format_recovery": j[i].get("grading_format_recovery")}
            for i, item in enumerate(original)])
    cal, mem, sel = [cohorts[x] for x in ("calibration", "memory", "selection")]
    cal_valid = paired(cal[1], cal[2]) & np.isfinite(scores([x["judge_score"] for x in cal[0]]))
    base_valid = paired(mem[1], [x["judge_score"] for x in mem[0]])
    shared_valid = np.isfinite(mem[2][base_valid])
    if cal_valid.sum() < 2 or (not cal_valid.all() and cal[2][cal_valid].std() < config["scoring"]["minimum_std"]) or shared_valid.sum() < base_memory.k:
        unavailable(output, "prepared_30b", "insufficient valid teacher calibration or memory grades",
                    calibration_pairs=int(cal_valid.sum()), memory_pairs=int(shared_valid.sum()))
        return None
    # Keep proxy normalization frozen; estimate teacher statistics from valid calibration labels.
    teacher_grades = cal[2][cal_valid]
    if teacher_grades.std() < config["scoring"]["minimum_std"]:
        raise ValueError("Degenerate teacher calibration.")
    norm = Normalization(base_norm.proxy_mean, base_norm.proxy_std,
                         float(teacher_grades.mean()), float(teacher_grades.std()), 0)
    norm.threshold = float(np.quantile(norm.gap(cal[1][cal_valid], teacher_grades), config["knn"]["gap_quantile"]))
    shared_base = GapMemory(base_memory.embeddings[shared_valid], base_memory.gaps[shared_valid],
                           base_memory.group_ids[shared_valid], base_memory.k, base_memory.temperature,
                           base_memory.encoder_identity)
    memory = GapMemory(shared_base.embeddings.copy(), norm.gap(mem[1][base_valid][shared_valid], mem[2][base_valid][shared_valid]),
                       shared_base.group_ids.copy(), base_memory.k, base_memory.temperature,
                       base_memory.encoder_identity)
    assert_matched(base_norm, shared_base, norm, memory)
    norm.save(folder / "normalization.json")
    memory.save(folder / "memory_initial.npz")
    rows = annotated(sel[0], sel[1], sel[2], sel[3], norm, memory, proxy.identity, sel[4], sel[5])
    for row in rows:
        row["diagnostic_teacher"] = "judge30b"
        row["diagnostic_memory_teacher"] = "judge30b"
    write_jsonl(folder / "selection_scored.jsonl", rows)
    atomic_json(folder / "selection_metrics.json", {
        **summarize_rows(rows, norm.threshold), "diagnostic_teacher": "judge30b",
        "note": "Same k and temperature as the 4B memory; no 30B-specific hyperparameter search."})
    names = ["normalization.json", "memory_initial.npz", "selection_metrics.json",
             "selection_scored.jsonl", *[f"{x}_raw.jsonl" for x in cohorts]]
    atomic_json(folder / "complete.json", {
        "teacher_model": config["models"]["judge30b"], "teacher_revision": resolved["judge30b"],
        "teacher_scorer_identity": teacher.identity, "encoder_identity": proxy.identity,
        "shared_response_hashes": hashes, "n_memory": len(memory.gaps),
        "shared_memory_indices": np.flatnonzero(shared_valid).tolist(),
        "excluded_memory_labels": int((~shared_valid).sum()),
        "k": memory.k, "temperature": memory.temperature,
        "artifact_sha256": {name: file_sha(folder / name) for name in names},
        "changes": "teacher grades, teacher normalization, normalized gap labels"})
    return norm, memory


def evaluate_teacher30b(proxy, teacher, norm, memory, output, arms, target):
    """Compare every policy using the same 30B grader, without regenerating answers."""
    from .run import annotated
    from .metrics import summarize_rows

    for arm in ["base", *arms]:
        step = 0 if arm == "base" else target
        folder = Path(output) / "evaluations" / "final" / arm / f"step_{step:06d}"
        destination = folder / "teacher30b_metrics.json"
        if destination.exists():
            continue
        status(output, "evaluate_final_30b", arm=arm, update=step)
        items = read_jsonl(folder / "responses.jsonl")
        if not items:
            raise ValueError(f"Final answers are missing for {arm}.")
        p = proxy.score(items, f"final30b/{arm}/proxy_cache")
        j = teacher.score(items, f"final30b/{arm}")
        rows = annotated(items, [x["score"] for x in p], [x["score"] for x in j],
                         np.stack([x["embedding"] for x in p]), norm, memory, proxy.identity, p, j)
        for row in rows:
            row["diagnostic_teacher"] = "judge30b"
            row["diagnostic_memory_teacher"] = "judge30b"
        write_jsonl(folder / "responses_teacher30b.jsonl", rows)
        atomic_json(destination, {**summarize_rows(rows, norm.threshold), "arm": arm,
                    "update": step, "cohort": "final", "diagnostic_teacher": "judge30b",
                    "diagnostic_memory_teacher": "judge30b", "responses_regenerated": 0,
                    "teacher_scorer_identity": teacher.identity})
