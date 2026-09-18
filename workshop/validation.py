"""CPU validation of saved gap predictions, with selection-locked cutoffs.

Label cutoffs define the diagnostic target; prediction cutoffs choose a review
operating point. Neither cutoff changes continuous regression metrics or PPO.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from .common import atomic_json, digest, read_json, read_jsonl, write_csv as csv_file
from .metrics import auroc, safe_corr, optimistic_tail_bias, TAIL_BIAS_KEYS, TAIL_BIAS_PROTOCOL

VERSION = "gap_validation_v3"
DEFAULTS = {"label_quantiles": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            "minimum_class_examples": 20, "minimum_class_questions": 20,
            "decision_metric": "balanced_accuracy"}
TEACHERS = {"judge": "prepared", "judge30b": "prepared_30b"}


def options(config):
    result = DEFAULTS | config.get("validation", {})
    if set(result) != set(DEFAULTS):
        raise ValueError("Unknown validation setting")
    if not result["label_quantiles"] or any(not 0 < q < 1 for q in result["label_quantiles"]):
        raise ValueError("Validation quantiles must be between zero and one")
    for key in ("minimum_class_examples", "minimum_class_questions"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"validation.{key} must be a positive integer")
    if result["decision_metric"] not in ("balanced_accuracy", "f1"):
        raise ValueError("validation.decision_metric must be balanced_accuracy or f1")
    return result


def usable(rows, keys):
    return [r for r in rows if all(isinstance(r.get(k), (int, float)) and np.isfinite(r[k]) for k in keys)]


def label(gaps, definition):
    if definition["comparator"] == ">=":
        return gaps >= definition["threshold"]
    return gaps > definition["threshold"]


def regression(gaps, predictions):
    n = len(gaps)
    mse = float(np.mean((gaps - predictions) ** 2)) if n else None
    variance = float(np.var(gaps)) if n else None
    return {"gap_mse": mse, "gap_rmse": float(np.sqrt(mse)) if mse is not None else None,
            "gap_mae": float(np.mean(np.abs(gaps - predictions))) if n else None,
            "gap_r2": 1 - mse / variance if n > 1 and variance > 0 else None,
            "gap_pearson": safe_corr(gaps, predictions),
            "gap_spearman": safe_corr(rankdata(gaps), rankdata(predictions)),
            "zero_gap_baseline_mse": float(np.mean(gaps ** 2)) if n else None,
            "evaluation_mean_baseline_mse": variance,
            "r2_note": "1 - SSE/SST against this cohort's mean; undefined for constant targets or fewer than two pairs"}


def confusion(y, detected):
    tp, fp = int((y & detected).sum()), int((~y & detected).sum())
    fn, tn = int((y & ~detected).sum()), int((~y & ~detected).sum())
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": recall, "specificity": specificity,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "flagged_fraction": float(detected.mean()) if len(detected) else None}


def decision_threshold(y, scores, objective):
    if not y.any() or y.all():
        return None, []
    candidates = []
    for t in np.unique(scores):
        candidates.append({"threshold": float(t), "comparator": ">=", **confusion(y, scores >= t)})
    winner = max(candidates, key=lambda x: (x[objective], x["threshold"]))
    return {"threshold": winner["threshold"], "comparator": ">=", "objective": objective,
            "selection_objective_value": winner[objective]}, candidates


def fit_thresholds(calibration_gaps, rows, settings):
    """Only calibration and selection may reach this function; no final input."""
    valid = usable(rows, ("gap", "predicted_gap"))
    gaps = np.array([r["gap"] for r in valid], float)
    scores = np.array([r["predicted_gap"] for r in valid], float)
    ids = np.array([r["id"] for r in valid])
    candidates = []
    if len(calibration_gaps):
        definitions = [(q, ">", float(np.quantile(calibration_gaps, q))) for q in settings["label_quantiles"]]
        definitions += [(None, ">", float(t)) for t in sorted(set(calibration_gaps) | {0.0})]
        definitions += [(0.95, ">=", float(np.quantile(calibration_gaps, 0.95)))]
        for q, op, threshold in definitions:
            definition = {"quantile": q, "threshold": threshold, "comparator": op}
            y, cal_y = label(gaps, definition), label(calibration_gaps, definition)
            positive_groups, negative_groups = len(set(ids[y])), len(set(ids[~y]))
            upper_tail = threshold >= 0 and 0 < cal_y.sum() <= len(cal_y) / 2
            supported = (min(int(y.sum()), int((~y).sum())) >= settings["minimum_class_examples"]
                         and min(positive_groups, negative_groups) >= settings["minimum_class_questions"])
            candidates.append({**definition, "calibration_positive": int(cal_y.sum()),
                               "selection_positive": int(y.sum()), "selection_negative": int((~y).sum()),
                               "positive_questions": positive_groups, "negative_questions": negative_groups,
                               "upper_tail": bool(upper_tail), "supported": supported,
                               "auroc": auroc(y, scores)})
    eligible = [r for r in candidates if r["supported"] and r["upper_tail"] and r["auroc"] is not None]
    if not eligible:
        return {"status": "unavailable", "reason": "No upper-tail label definition with sufficient class support",
                "label_definition": None, "prediction_cutoff": None,
                "label_candidates": candidates, "decision_candidates": []}
    best = max(eligible, key=lambda r: (r["auroc"], r["comparator"] == ">", r["quantile"] or -1))
    definition = {k: best[k] for k in ("quantile", "threshold", "comparator")}
    cutoff, decisions = decision_threshold(label(gaps, definition), scores, settings["decision_metric"])
    return {"status": "selected", "label_definition": definition, "prediction_cutoff": cutoff,
            "label_candidates": candidates, "decision_candidates": decisions}


def reward_alignment(rows):
    """Rank correct answers using the reward PPO receives, with matched rows.

    These labels come from answer verification, independently of the grader gap
    or any selected cutoff. Keep strict formatting and numeric matching separate.
    """
    result = {}
    for target, prefix in (("correct", "correctness"), ("numeric_match", "numeric_correctness")):
        valid = [r for r in usable(rows, ("proxy_z", "predicted_gap")) if type(r.get(target)) is bool]
        y = np.array([r[target] for r in valid], bool)
        proxy = np.array([r["proxy_z"] for r in valid], float)
        corrected = proxy - np.array([r["predicted_gap"] for r in valid], float)
        judged = usable(valid, ("judge_z",))
        values = {"n": len(valid), "excluded": len(rows) - len(valid), "positive": int(y.sum()),
                  "proxy_auroc": auroc(y, proxy), "corrected_reward_auroc": auroc(y, corrected),
                  "judge_n": len(judged),
                  "judge_auroc": auroc([r[target] for r in judged], [r["judge_z"] for r in judged])}
        result.update({f"{prefix}_{key}": value for key, value in values.items()})
    return result


def evaluate_rows(rows, locked):
    valid = usable(rows, ("gap", "predicted_gap"))
    gaps, scores = (np.array([r[k] for r in valid], float) for k in ("gap", "predicted_gap"))
    metrics = {"n": len(rows), "n_scored": len(valid), "n_unscored": len(rows) - len(valid),
               "questions_scored": len({r["id"] for r in valid}), **regression(gaps, scores),
               **optimistic_tail_bias([r.get('gap') for r in rows], [r.get('predicted_gap') for r in rows], [r['id'] for r in rows]),
               **reward_alignment(rows)}
    mean = locked.get("calibration_mean_gap")
    metrics["calibration_mean_baseline_mse"] = float(np.mean((gaps - mean) ** 2)) if len(gaps) and mean is not None else None
    reward_rows = usable(valid, ("judge_z", "proxy_z"))
    j = np.array([r["judge_z"] for r in reward_rows], float)
    corrected = np.array([r["proxy_z"] - r["predicted_gap"] for r in reward_rows], float)
    reward = regression(j, corrected)
    metrics.update(corrected_judge_n=len(j), corrected_judge_mse=reward["gap_mse"],
                   corrected_judge_r2=reward["gap_r2"])
    definition, cutoff = locked["label_definition"], locked["prediction_cutoff"]
    if definition is None:
        metrics.update(high_gap_auroc=None, high_gap_average_precision=None, high_gap_count=None,
                       high_gap_rate=None, classification_status="unavailable: no supported validation target")
        return metrics
    y = label(gaps, definition)
    metrics.update(high_gap_count=int(y.sum()), high_gap_rate=float(y.mean()) if len(y) else None,
                   high_gap_auroc=auroc(y, scores),
                   high_gap_average_precision=float(average_precision_score(y, scores)) if y.any() else None,
                   classification_status="available" if y.any() and not y.all() else "AUROC unavailable: fewer than two classes")
    if cutoff is not None:
        metrics.update(confusion(y, scores >= cutoff["threshold"]))
    return metrics


def intervals(rows, locked, samples, seed):
    valid = usable(rows, ("gap", "predicted_gap"))
    ids = np.array([r["id"] for r in valid])
    groups = [np.flatnonzero(ids == key) for key in sorted(set(ids))]
    values = {key: [] for key in ("gap_mse", "gap_r2", "high_gap_auroc")}
    if not groups:
        return {k: {"ci95": None, "valid_resamples": 0} for k in values}
    gaps, scores = (np.array([r[k] for r in valid], float) for k in ("gap", "predicted_gap"))
    y = label(gaps, locked["label_definition"]) if locked["label_definition"] else None
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        ix = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        mse, variance = float(np.mean((gaps[ix] - scores[ix]) ** 2)), float(np.var(gaps[ix]))
        values["gap_mse"].append(mse)
        if len(ix) > 1 and variance > 0:
            values["gap_r2"].append(1 - mse / variance)
        auc = auroc(y[ix], scores[ix]) if y is not None else None
        if auc is not None:
            values["high_gap_auroc"].append(auc)
    return {k: {"ci95": np.quantile(v, [0.025, 0.975]).tolist() if v else None,
                "valid_resamples": len(v)} for k, v in values.items()}


def validate_saved_run(output, destination=None, *, include_final=True, config=None):
    output = Path(output).resolve()
    if not output.is_dir():
        raise ValueError(f"Run directory does not exist: {output}")
    destination = Path(destination).resolve() if destination else output / "validation" / VERSION
    # A dedicated report directory protects original scores, normalization and reports.
    if destination == output or destination in output.parents:
        raise ValueError("Use a separate validation report directory")
    if destination.is_relative_to(output) and not destination.is_relative_to(output / "validation"):
        raise ValueError("Within a run, validation reports must be under validation/")
    config = config if config is not None else read_json(output / "config.json")
    settings = options(config)
    samples, seed = config["evaluation"]["bootstrap_samples"], config["seed"]
    if type(samples) is not int or samples < 0:
        raise ValueError("evaluation.bootstrap_samples must be a nonnegative integer")
    destination.mkdir(parents=True, exist_ok=True)
    report = {"version": VERSION, "scope": "diagnostics only; no training or score changes",
              "optimistic_tail_bias_protocol": TAIL_BIAS_PROTOCOL,
              "seed": seed, "settings": settings, "teachers": {}, "final": [],
              "bootstrap": {"samples": samples, "seed": seed, "unit": "question, retaining all answers",
                            "scope": "conditional on fitted predictor, calibration and selected cutoffs; no tuning or training-seed adjustment"}}
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__), Path(__file__).with_name("metrics.py")]}
    for teacher, folder in TEACHERS.items():
        paths = [output / folder / name for name in
                 ("normalization.json", "calibration_raw.jsonl", "memory_raw.jsonl", "selection_scored.jsonl")]
        if not all(p.is_file() for p in paths):
            report["teachers"][teacher] = {"status": "unavailable", "reason": "Preparation files are not available"}
            continue
        norm = read_json(paths[0])
        calibration, memory, selection = (read_jsonl(p) for p in paths[1:])
        cohorts = [set(r["id"] for r in rows) for rows in (calibration, memory, selection)]
        if any(cohorts[i] & cohorts[j] for i in range(3) for j in range(i)):
            raise ValueError("Calibration, memory and validation question IDs overlap")
        if min(norm["proxy_std"], norm["judge_std"]) <= 0:
            raise ValueError("Invalid saved normalization")
        def gap(row):
            return ((row["proxy_score"] - norm["proxy_mean"]) / norm["proxy_std"]
                    - (row["judge_score"] - norm["judge_mean"]) / norm["judge_std"])
        cal = np.array([gap(r) for r in usable(calibration, ("proxy_score", "judge_score"))], float)
        for row in usable(selection, ("gap", "proxy_score", "judge_score")):
            if not np.isclose(row["gap"], gap(row), rtol=0, atol=1e-10):
                raise ValueError("Saved validation gap differs from its normalization")
        if any(r.get("diagnostic_teacher", teacher) != teacher or r.get("diagnostic_memory_teacher", teacher) != teacher for r in selection):
            raise ValueError("Validation teacher and predictor identity mismatch")
        identity = {"version": VERSION, "teacher": teacher, "settings": settings,
                    "bootstrap_samples": samples, "seed": seed, "analysis_source_sha256": source_hashes,
                    "input_sha256": {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
        lock_path = destination / f"{teacher}_selection.json"
        if lock_path.exists():
            locked = read_json(lock_path)
            if locked["identity"] != identity:
                raise ValueError(f"Validation inputs, settings or source changed; choose a new --destination: {lock_path}")
        else:
            locked = {"identity": identity, **fit_thresholds(cal, selection, settings),
                      "calibration_mean_gap": float(cal.mean()) if len(cal) else None,
                      "excluded_calibration_answers": len(calibration) - len(cal),
                      "cohort_questions": {k: len(ids) for k, ids in zip(("calibration", "memory", "selection"), cohorts)},
                      "posthoc": (output / "final_protocol.json").exists() or any((output / "evaluations/final").glob("*/step_*/responses*.jsonl")),
                      "selection_note": "Tuning diagnostics, not independent performance evidence. MSE/R2 use continuous predictions and are not threshold objectives."}
            locked["metrics"] = evaluate_rows(selection, locked)
            locked["intervals"] = intervals(selection, locked, samples, seed)
            # Save the choice before final responses can be opened.
            atomic_json(lock_path, locked)
        csv_file(destination / f"{teacher}_label_candidates.csv", locked["label_candidates"])
        csv_file(destination / f"{teacher}_decision_candidates.csv", locked["decision_candidates"])
        report["teachers"][teacher] = locked
        if not include_final:
            continue
        filename = "responses.jsonl" if teacher == "judge" else "responses_teacher30b.jsonl"
        for path in sorted((output / "evaluations/final").glob(f"*/step_*/{filename}")):
            rows = read_jsonl(path)
            if set(r["id"] for r in rows) & set.union(*cohorts):
                raise ValueError("Final questions overlap preparation/validation")
            if any(r.get("diagnostic_teacher", teacher) != teacher or r.get("diagnostic_memory_teacher", teacher) != teacher for r in rows):
                raise ValueError("Final diagnostic teacher and predictor identity mismatch")
            # Versioned cached reports remain separate from original metrics.json.
            key = {"selection_identity": digest(identity), "response_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            final_path = destination / f"{teacher}_{path.parent.parent.name}_{path.parent.name}.json"
            if final_path.exists() and read_json(final_path).get("identity") == key:
                final = read_json(final_path)
            else:
                final = {"identity": key, "teacher": teacher, "policy": path.parent.parent.name,
                         "step": int(path.parent.name.removeprefix("step_")), "source": path.relative_to(output).as_posix(),
                         "metrics": evaluate_rows(rows, locked), "intervals": intervals(rows, locked, samples, seed)}
                atomic_json(final_path, final)
            report["final"].append(final)
    atomic_json(destination / "summary.json", report)
    rows = [{"cohort": "selection", "teacher": t, "policy": "preparation", **r["metrics"]}
            for t, r in report["teachers"].items() if "metrics" in r]
    rows += [{"cohort": "final", "teacher": r["teacher"], "policy": r["policy"], "step": r["step"], **r["metrics"]} for r in report["final"]]
    csv_file(destination / "metrics.csv", rows)
    def fmt(value):
        return "unavailable" if value is None else f"{value:.4f}"
    lines = ["# GSM8K gap and reward validation", "", "Selection chooses gap cutoffs; final evaluation uses those fixed cutoffs. "
             "Reward/correctness AUROC separately measures how well rewards rank correct answers. "
             "Neither AUROC is policy-answer accuracy. Original run artifacts are unchanged.", ""]
    for teacher, locked in report["teachers"].items():
        lines += [f"## {teacher}", ""]
        if "metrics" not in locked:
            lines += [locked["reason"], ""]
            continue
        lines += [f"Label definition: `{locked['label_definition']}`.", "",
                  f"Prediction cutoff: `{locked['prediction_cutoff']}`.", "",
                  f"Post hoc on a run with final artifacts already present: **{locked['posthoc']}**.", ""]
    lines += ["| Cohort | Teacher | Policy | Scored / all | High-gap AUROC | High-gap AP | Gap MSE | RMSE | MAE | Gap R2 | Corrected-judge R2 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['cohort']} | {r['teacher']} | {r['policy']} | {r['n_scored']} / {r['n']} | "
                     + " | ".join(fmt(r.get(k)) for k in ("high_gap_auroc", "high_gap_average_precision", "gap_mse",
                                                          "gap_rmse", "gap_mae", "gap_r2", "corrected_judge_r2")) + " |")
    lines += ['', '## Optimistic Tail Bias', '',
              'Mean(actual gap - predicted gap) in each predictor/cohort lower tail. Positive values mean '
              'excess reward under signed correction. All cutoff ties are retained; n is scored / selected answers. '
              'Missing actual gaps do not move the prediction cutoff. Small tails have limited support; '
              'no tail confidence intervals are claimed.', '',
              '| Cohort | Teacher | Policy | OTB 1% (n) | OTB 5% (n) | OTB 10% (n) |',
              '|---|---|---|---:|---:|---:|']
    for r in rows:
        lines.append('| '+' | '.join([r['cohort'], r['teacher'], r['policy'],
            *[f"{fmt(r[k])} ({r[k+'_n']}/{r[k+'_selected']})" for k in TAIL_BIAS_KEYS]])+' |')
    lines += ["", "## Reward alignment with answer correctness", "",
              "All proxy/corrected comparisons below use the same answers. Higher reward predicts a correct answer; "
              "the corrected reward is proxy_z - predicted_gap. No gap threshold is involved. "
              "Strict correctness includes the required answer format; numeric matching is reported separately. "
              "These are descriptive ranking metrics, not an estimate of accuracy after another PPO run.", "",
              "| Cohort | Teacher | Policy | Label | Scored / all | Correct | Proxy AUROC | Corrected-reward AUROC | Judge AUROC | Judge n |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        for prefix, name in (("correctness", "Strict correctness"), ("numeric_correctness", "Numeric match")):
            lines.append(f"| {r['cohort']} | {r['teacher']} | {r['policy']} | {name} | "
                         f"{r[prefix + '_n']} / {r['n']} | {r[prefix + '_positive']} | "
                         + " | ".join(fmt(r.get(prefix + '_' + k)) for k in ("proxy_auroc", "corrected_reward_auroc", "judge_auroc"))
                         + f" | {r[prefix + '_judge_n']} |")
    lines += ["", "MSE, RMSE, MAE and R2 use continuous normalized gaps. R2 can be negative; it is not squared correlation. "
              "Corrected-judge R2 instead compares proxy_z - predicted_gap with judge_z. AUROC/AP use continuous predicted_gap; "
              "the separate prediction cutoff determines precision/recall/F1 and balanced accuracy.", "",
              "Missing/nonfinite pairs are excluded and counted. Single-class AUROC, constant-target R2 and unsupported threshold "
              "searches remain unavailable. No fabricated labels are used. Lack of diagnostic support does not stop training.", "",
              "Intervals in summary.json resample whole questions conditional on the fitted predictor and chosen cutoffs. "
              "They do not account for threshold selection, calibration uncertainty or training-seed variability. "
              "Selection values are tuning diagnostics. The final benchmark may already have been inspected.", "",
              "k and temperature remain those of the saved predictor. Existing kNN grid selection minimizes validation MSE; "
              "on one fixed target cohort this also maximizes R2. Threshold search does not refit that predictor.", ""]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Validation report: {destination / 'report.md'}", flush=True)
    return report


def compare_predictors(output, target=None):
    """Compare frozen 4B kNN and ridge on the very same saved answers/features."""
    import gzip
    import json
    from .memory import GapMemory, Normalization, RidgeGap, load_features, corrected_reward
    from .grading import number
    output = Path(output)
    bank = output/'prepared/memory_matched.npz'
    if not bank.exists():
        return
    config = read_json(output/'config.json')
    norm = Normalization.load(output/'prepared/normalization.json')
    memory = GapMemory.load(bank)
    predictors = {'knn_static': memory}
    if (output/'prepared/ridge/selection.json').exists():
        predictors['ridge'] = RidgeGap.load(output/'prepared/ridge', memory)
    lock_path = output/'validation'/VERSION/'judge_selection.json'
    if not lock_path.exists():
        return
    locked = read_json(lock_path)
    definition = locked['label_definition']
    cohorts = [('selection', output/'prepared/selection_scored.jsonl', output/'prepared/selection_features.npz')]
    kind = 'final' if (output/'final_protocol.json').exists() else 'monitor'
    for folder in sorted((output/'evaluations'/kind).glob('*/step_*')):
        step = int(folder.name.removeprefix('step_'))
        if (target is None or folder.parent.name == 'base' or step == target) and (folder/'metrics.json').exists():
            cohorts.append((kind+'/'+folder.parent.name+f'/{step}', folder/'responses.jsonl', folder/'features.npz'))
    dest = output/'predictors'
    dest.mkdir(exist_ok=True)
    metrics = []
    with gzip.open(dest/'all_predictions.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for cohort, responses, features in cohorts:
            rows = read_jsonl(responses)
            if not rows:
                continue
            x = load_features(features, rows, memory.encoder_identity)
            gaps = norm.gap([r['proxy_score'] for r in rows], [r['judge_score'] for r in rows])
            valid = np.isfinite(gaps)
            high = label(gaps[valid], definition) if definition else None
            answer_policy = None if cohort == 'selection' else cohort.split('/')[1]
            def accuracy(key):
                return float(np.mean([r[key] for r in rows])) if answer_policy is not None and all(type(r.get(key)) is bool for r in rows) else None
            for name, predictor in predictors.items():
                predicted = predictor.predict(x, [r['id'] for r in rows], memory.encoder_identity)[0]
                tails = optimistic_tail_bias(gaps, predicted, [r['id'] for r in rows])
                metrics.append({'seed': config['seed'], 'cohort': cohort, 'predictor': name,
                    'answers': len(rows), 'valid_pairs': int(valid.sum()),
                    'answer_policy': answer_policy, 'answer_policy_accuracy': accuracy('correct'),
                    'answer_policy_numeric_accuracy': accuracy('numeric_match'),
                    'selection_diagnostic': cohort == 'selection',
                    **regression(gaps[valid], predicted[valid]),
                    **tails,
                    'high_gap_auroc': auroc(high, predicted[valid]) if high is not None else None,
                    'high_gap_average_precision': float(average_precision_score(high, predicted[valid])) if high is not None and high.any() else None})
                rewards = corrected_reward(norm.proxy_z([r['proxy_score'] for r in rows]), predicted, config['knn']['correction'])
                for i, row in enumerate(rows):
                    record = {**row, 'seed': config['seed'], 'cohort': cohort, 'predictor': name,
                              'gap': number(gaps[i]), 'predicted_gap': number(predicted[i]),
                              'corrected_reward': number(rewards[i]),
                              'gap_residual': number(gaps[i]-predicted[i]),
                              **{key+'_member': bool(predicted[i] <= tails[key+'_cutoff']) if np.isfinite(predicted[i]) and tails[key+'_cutoff'] is not None else None for key in TAIL_BIAS_KEYS},
                              'high_gap': bool(label(gaps[i:i+1], definition)[0]) if valid[i] and definition else None}
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
    csv_file(dest/'metrics.csv', metrics)
    atomic_json(dest/'summary.json', {'seed': config['seed'], 'label_definition': definition,
        'target': 'actual normalized 4B proxy-judge gap', 'metrics': metrics,
        'ridge_alpha_selection': 'separate validation MSE; never refit on validation/final',
        'high_gap_target_selection': 'same existing kNN calibration/validation rule for both predictors',
        'optimistic_tail_bias_protocol': TAIL_BIAS_PROTOCOL,
        'new_judge_calls': 0})
    fmt = lambda value: 'unavailable' if value is None else f'{value:.4f}'
    lines = ['# Matched kNN and ridge gap prediction', '',
        'Both predictors use the same 4B memory labels and are evaluated on identical answers. '
        'Ridge alpha minimizes separate validation MSE. The shared high-gap target is selected '
        'by the kNN validation rule before PPO/final evaluation; it can favor kNN on validation. '
        'Selection rows are tuning diagnostics. Thresholds do not change continuous regression metrics or PPO rewards.', '',
        'Optimistic Tail Bias (OTB) is mean(actual gap - predicted gap) in the lowest 1%, 5%, or 10% '
        'of predicted gaps. Cutoffs use all finite predictions in each cohort; ties are included. '
        'Only scored pairs contribute to the bias; parentheses show scored / selected tail answers. '
        'A positive bias means excess corrected reward under signed correction, for these answers. '
        'It is not proof of a causal effect on PPO accuracy, and this tail need not be the highest total rewards.', '',
        '| Answers from | Predictor | Pairs / all | MSE | R2 | Pearson | Spearman | AUROC | AP | OTB 1% (n) | OTB 5% (n) | OTB 10% (n) | Answer-policy strict accuracy |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in metrics:
        lines.append('| '+' | '.join([row['cohort'], row['predictor'], f"{row['valid_pairs']} / {row['answers']}",
            *[fmt(row[k]) for k in ('gap_mse', 'gap_r2', 'gap_pearson', 'gap_spearman', 'high_gap_auroc', 'high_gap_average_precision')],
            *[f"{fmt(row[k])} ({row[k+'_n']}/{row[k+'_selected']})" for k in TAIL_BIAS_KEYS],
            f"{row['answer_policy_accuracy']:.2%}" if row['answer_policy_accuracy'] is not None else 'not a PPO evaluation'])+' |')
    lines += ['', 'Accuracy belongs to the policy named in Answers from, not necessarily the predictor in that row. '
        'Both predictors see identical answers for each cohort. See ../report.md for shared-base predictor diagnostics '
        'next to each corresponding trained policy accuracy. Selection rows are tuning diagnostics.', '',
        'CSV also includes RMSE, MAE, Pearson, Spearman, AP, numeric accuracy, tail cutoffs, distinct question counts, '
        'selected fractions and missing-label counts. Small tails can contain only one scored answer; all tied predictions '
        'can make a tail much larger than its nominal percentage. No tail confidence intervals are claimed.', '',
        '`all_predictions.jsonl.gz` contains every answer, actual gap, both predictions, corrected rewards, signed '
        'residuals and tail-membership flags. Missing grades remain missing. No model refitting or new judge calls are needed.', '']
    (dest/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Saved run to validate; no models are loaded")
    parser.add_argument("--destination", type=Path, help="Separate validation report directory")
    args = parser.parse_args(argv)
    try:
        validate_saved_run(args.output, args.destination)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
