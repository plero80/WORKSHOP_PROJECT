from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import random
import re
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs/default.yaml"
OUTPUT_ROOT = ROOT / "outputs"
ENGINE_ID = "workshop.ppo.v1"
ARMS = ("proxy", "judge", "knn_static", "knn_static_30b", "ridge", "knn_refresh", "oracle")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def response_key(encoder_identity, row):
    return digest([encoder_identity, row['question'], row['reference'], row['response']])


def seed_for(*parts):
    """Stable arm-independent minibatch randomness, matching the original PPO rule."""
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2**31 - 1)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def group_id(question: str) -> str:
    text = unicodedata.normalize("NFKC", question)
    return digest(re.sub(r"\s+", " ", text).strip().casefold())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A killed process may leave one torn final line. Remove only that tail,
    # preserving all complete events before appending the next record.
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                size = f.tell()
                position = size
                while position:
                    start = max(0, position - 4096)
                    f.seek(start)
                    block = f.read(position - start)
                    index = block.rfind(b"\n")
                    if index >= 0:
                        f.truncate(start + index + 1)
                        break
                    position = start
                else:
                    f.truncate(0)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    if not line.endswith("\n"):
                        break  # interrupted append at EOF; not a complete event
                    raise
    return result


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite(values, name):
    if not np.isfinite(np.asarray(values)).all():
        raise ValueError(f"Non-finite {name}; refusing to use invalid numerical values.")


def load_config(path):
    path = Path(path)
    if path.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        c = yaml.safe_load(path.read_text(encoding='utf-8'))
    else:
        c = read_json(path)
    return validate_config(c)


def validate_config(c):
    if not isinstance(c, dict):
        raise ValueError('Configuration must be a mapping.')
    from .validation import options as validation_options
    validation_options(c)
    if c["generation"]["temperature"] != 1.0:
        raise ValueError("PPO requires generation.temperature = 1.0.")
    d, p, g, s, k = (c[x] for x in ("dataset", "ppo", "generation", "scoring", "knn"))
    if not isinstance(c["runtime"].get("gradient_checkpointing", True), bool):
        raise ValueError("runtime.gradient_checkpointing must be a JSON boolean.")
    if min(g["batch_size"], s["batch_size"]) < 1:
        raise ValueError("Generation and grading batch sizes must be positive.")
    for key in ("ppo_microbatch_size",):
        value = c["runtime"].get(key, 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"runtime.{key} must be a positive integer.")
    if not isinstance(c["seed"], int):
        raise ValueError("This runner uses exactly one integer seed.")
    if p["prompts_per_update"] * p["responses_per_prompt"] < p["minibatch_size"]:
        raise ValueError("PPO minibatch cannot exceed rollout size.")
    if p["minibatch_size"] < 2 or p["epochs"] < 1:
        raise ValueError("Need at least two responses per PPO minibatch and one epoch.")
    if not 0 < g["temperature"] <= 2 or min(g["max_new_tokens"], g["max_prompt_tokens"]) < 1:
        raise ValueError("Invalid generation limits or temperature.")
    if p["gamma"] != 1.0 or not 0 <= p["gae_lambda"] <= 1:
        raise ValueError("This finite-horizon experiment fixes gamma=1.")
    if min(d[x] for x in ("calibration", "memory", "selection", "monitor", "ppo", "final")) < 1:
        raise ValueError("All main cohorts must be nonempty.")
    if k["correction"] not in ("signed", "positive_only"):
        raise ValueError("Choose signed or positive_only correction.")
    if not 0 < k["gap_quantile"] < 1 or min(k["k_grid"]) < 1 or min(k["temperature_grid"]) <= 0:
        raise ValueError("Invalid kNN grid.")
    if s["mode"] not in ("rationale_then_score", "expected_digit"):
        raise ValueError("Unknown scoring mode.")
    if s["retry_max_new_tokens"] < s["max_new_tokens"]:
        raise ValueError("The score retry must allow at least as many tokens.")
    if not c['arms'] or len(set(c["arms"])) != len(c["arms"]) or not set(c["arms"]) <= set(ARMS):
        raise ValueError("Unknown or duplicate arm.")
    if 'ridge' in c['arms']:
        ridge = c.get('ridge', {})
        alphas = ridge.get('alphas', [])
        if (not isinstance(alphas, list) or not alphas or
            any(type(a) not in (int, float) or not np.isfinite(a) or a <= 0 for a in alphas) or
            len(set(alphas)) != len(alphas)):
            raise ValueError('ridge.alphas must contain distinct finite positive numbers.')
        if type(ridge.get('cpu_threads')) is not int or ridge['cpu_threads'] < 1:
            raise ValueError('ridge.cpu_threads must be a positive integer.')
    for key in ('pilot_updates', 'full_updates', 'prompts_per_update', 'responses_per_prompt',
                'epochs', 'minibatch_size', 'checkpoint_every', 'monitor_every'):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f'ppo.{key} must be a positive integer.')
    if c["runtime"]["dtype"] not in ("bfloat16", "float32"):
        raise ValueError("Use bfloat16 on supported GPUs or float32; fp16 without a scaler is not supported.")
    for value in c.get("completion_reward", {}).values():
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError("Completion penalties must be finite nonnegative numbers.")
    if "knn_static_30b" in c["arms"]:
        if "judge30b" not in c["models"] or "judge30b" not in c["revisions"]:
            raise ValueError("The new arm requires a judge30b model and revision.")
        teacher = c.get("teacher30b", {})
        if not isinstance(teacher.get("batch_size"), int) or teacher["batch_size"] < 1:
            raise ValueError("teacher30b.batch_size must be positive.")
        if not isinstance(teacher.get("evaluate_final"), bool):
            raise ValueError("teacher30b.evaluate_final must be boolean.")
    return c


@contextlib.contextmanager
def run_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+b") as f:
        if os.name == "nt":
            import msvcrt
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                f.write(b"0")
                f.flush()
            f.seek(0)
            lock = lambda: msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: (f.seek(0), msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            lock = lambda: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(f, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as e:
            raise RuntimeError(f"An experiment already holds the lock for {output}.") from e
        try:
            yield
        finally:
            unlock()


def status(output, stage, **fields):
    message = {"stage": stage, "timestamp": time.time(), **fields}
    atomic_json(Path(output) / "status.json", message)
    print(json.dumps(message), flush=True)
