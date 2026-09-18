from __future__ import annotations

import random
from pathlib import Path

from .answers import verify_answer
from .common import atomic_json, digest, group_id, read_json


def partition_rows(train_rows, test_rows, sizes, seed):
    train, test = {}, {}
    duplicate_count = 0
    for source, source_rows, target in (("train", train_rows, train), ("test", test_rows, test)):
        for row in source_rows:
            q, a = row["question"], row["answer"]
            gid = group_id(q)
            verify_answer("", a)  # Validate all gold numbers, before spending GPU time.
            if gid in target:
                duplicate_count += 1
                if target[gid]["reference"] != a:
                    raise ValueError("Duplicate question has conflicting reference solutions.")
            else:
                target[gid] = {"id": gid, "question": q, "reference": a, "source": source}
    overlaps = set(train) & set(test)
    # Preserve the official test set. Exclude overlapping training questions.
    for gid in overlaps:
        del train[gid]
    rows = list(train.values())
    rng = random.Random(seed)
    rng.shuffle(rows)
    names = ("calibration", "memory", "selection", "monitor", "refresh", "ppo")
    required = sum(sizes[name] for name in names)
    if required > len(rows):
        raise ValueError(f"Need {required} distinct GSM8K training questions; only {len(rows)} are eligible. Reduce cohort sizes explicitly.")
    result, cursor = {}, 0
    for name in names:
        result[name] = rows[cursor:cursor + sizes[name]]
        cursor += sizes[name]
    final = list(test.values())
    rng.shuffle(final)
    if sizes["final"] > len(final):
        raise ValueError(f"Need {sizes['final']} final questions; only {len(final)} unique official test questions exist.")
    result["final"] = final[:sizes["final"]]
    seen = set()
    for name, items in result.items():
        ids = {x["id"] for x in items}
        assert len(ids) == len(items) and not ids & seen, f"Split leakage: {name}"
        seen |= ids
    return {"cohorts": result, "audit": {"deduplicated": duplicate_count,
             "train_test_overlap_excluded_from_train": len(overlaps), "unused_train": len(rows) - cursor,
             "sizes": {k: len(v) for k, v in result.items()}, "seed": seed}}


def prepare_data(config, output, resolved):
    path = Path(output) / "data" / "splits.json"
    if path.exists():
        return read_json(path)
    from datasets import load_dataset
    ds = load_dataset(config["dataset"]["id"], config["dataset"]["config"], revision=resolved["dataset"])
    split = partition_rows(ds["train"], ds["test"], config["dataset"], config.get("data_seed", config["seed"]))
    split["fingerprint"] = digest(split)
    split["hf_revision"] = resolved["dataset"]
    atomic_json(path, split)
    return split


def rollout_questions(rows, update, count, seed):
    # Explicit reproducible cycles over the whole PPO cohort. Shared by all arms.
    result = []
    start = update * count
    cached_cycle, order = None, None
    for i in range(start, start + count):
        cycle, index = divmod(i, len(rows))
        if cycle != cached_cycle:
            order = list(range(len(rows)))
            random.Random(seed + cycle).shuffle(order)
            cached_cycle = cycle
        result.append(rows[order[index]])
    return result
