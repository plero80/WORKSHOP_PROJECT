"""Missing grades are exclusions, never substitute rewards."""
from pathlib import Path

import numpy as np

from .common import atomic_json, read_json

PROTOCOL = "ungraded_review_v1"
RECOVERY_ID = 'grading_retry_v1'
RECOVERY_POLICY = {'id': RECOVERY_ID, 'extra_max_new_tokens': [640, 1280, 2560]}
FORMAT_ID = 'grading_format_v1'


def number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def scores(values):
    # NaN is only an internal mask. Persist missing values as JSON null.
    return np.asarray(values, dtype=float)


def paired(proxy, judge):
    return np.isfinite(scores(proxy)) & np.isfinite(scores(judge))


def subset(cohort, mask):
    items, ps, js, emb, p, j = cohort
    indices = np.flatnonzero(mask)
    return ([items[i] for i in indices], ps[mask], js[mask], emb[mask],
            [p[i] for i in indices], [j[i] for i in indices])


def review_case(output, key, role, stage, row, value, attempts, scorer_identity):
    relative = Path("review") / "ungraded" / f"{key}.json"
    path = Path(output) / relative
    previous = read_json(path) if path.exists() else {}
    atomic_json(path, {
        "protocol": PROTOCOL, "status": "unscored", "reason": "no_valid_grade_after_retries",
        "case_id": key, "role": role, "scorer_identity": scorer_identity,
        "stages": sorted(set(previous.get("stages", [])) | {stage}),
        "question_id": row["id"], "question": row["question"], "reference": row["reference"],
        "response": row["response"], "score": None,
        "judge_output": value["judge_output"], "attempts": attempts,
    })
    readme = Path(output) / "review" / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Ungraded examples\n\nEach file in `ungraded/` contains the question, reference, "
            "candidate answer and every failed grader reply for one scorer/answer pair.\n\n"
            "These examples have no reward. Training and memory fitting exclude missing labels; "
            "accuracy still counts all generated answers. Editing these files does not change "
            "the experiment or its score cache.\n", encoding="utf-8")
    return relative.as_posix()


def unavailable(output, component, reason, **counts):
    record = {"status": "unavailable", "reason": reason, "protocol": PROTOCOL, **counts}
    atomic_json(Path(output) / component / "unavailable.json", record)
    print(f"{component}: {reason}; continuing with available stages.", flush=True)
    return record
