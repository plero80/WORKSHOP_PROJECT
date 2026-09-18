"""CPU-only paper diagnostics from frozen predictors and saved evaluation answers.

No model downloads, refitting, generation, or grader calls. Monitoring trajectories
and final benchmark evaluations remain separate throughout the analysis.
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.stats import binomtest, kendalltau, rankdata
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from .common import atomic_json, file_sha, read_json, read_jsonl, seed_for, write_csv
from .grading import number
from .memory import GapMemory, Normalization, RidgeGap, corrected_reward, load_features
from .metrics import auroc, optimistic_tail_bias, safe_corr
from .validation import regression, label

VERSION = 'core_paper_analysis_v1'
TOP_FRACTIONS = (.01, .05, .10, .20)


def mean(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def column(rows, name):
    return np.asarray([r.get(name) for r in rows], float)


def optimism_statistics(values):
    raw = np.asarray(values, float)
    values = raw[np.isfinite(raw)]
    result = {'optimism_n': len(values), 'optimism_excluded': len(raw)-len(values)}
    functions = {'mean': np.mean, 'median': np.median, 'std': np.std, 'max': np.max,
                 'p90': lambda x: np.quantile(x, .9), 'p95': lambda x: np.quantile(x, .95),
                 'p99': lambda x: np.quantile(x, .99),
                 'fraction_gt_0': lambda x: np.mean(x > 0),
                 'fraction_gt_0_5': lambda x: np.mean(x > .5),
                 'fraction_gt_1': lambda x: np.mean(x > 1.)}
    return result | {'optimism_'+k: float(fn(values)) if len(values) else None for k, fn in functions.items()}


def top_tail(scores, fraction):
    """Inclusive quantile selection: ties may exceed the nominal percentage."""
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or not 0 < fraction <= 1:
        raise ValueError('Top-tail selection needs a vector and fraction in (0, 1].')
    valid = np.isfinite(scores)
    cutoff = float(np.quantile(scores[valid], 1-fraction, method='linear')) if valid.any() else None
    return valid & (scores >= cutoff) if cutoff is not None else valid, cutoff


def high_reward_tail_bias(optimism, rewards, ids):
    optimism, rewards, ids = np.asarray(optimism, float), np.asarray(rewards, float), np.asarray(ids)
    if optimism.ndim != 1 or rewards.shape != optimism.shape or ids.shape != optimism.shape:
        raise ValueError('Tail optimism requires aligned answers, rewards and IDs.')
    result = {}
    for fraction in TOP_FRACTIONS:
        key = f'high_reward_tail_optimism_{round(100*fraction):02d}'
        selected, cutoff = top_tail(rewards, fraction)
        scored = selected & np.isfinite(optimism)
        result.update({key: mean(optimism[scored]), key+'_cutoff': cutoff,
                       key+'_selected': int(selected.sum()), key+'_n': int(scored.sum()),
                       key+'_questions': len(set(ids[scored].tolist())),
                       key+'_unscored': int((selected & ~scored).sum()),
                       key+'_fraction': float(selected.sum()/np.isfinite(rewards).sum()) if np.isfinite(rewards).any() else None})
    return result


def ranking(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    supported = len(a) > 1 and np.ptp(a) > 0 and np.ptp(b) > 0
    return {'n': len(a), 'pearson': safe_corr(a, b) if supported else None,
            'spearman': safe_corr(rankdata(a), rankdata(b)) if supported else None,
            'kendall': float(kendalltau(a, b, variant='b').statistic) if supported else None}


def gap_statistics(gaps, predictions):
    """Reuse the regression implementation with explicit constant-series support."""
    result = regression(gaps, predictions)
    correlations = ranking(gaps, predictions)
    result.update({f'gap_{key}': correlations[key] for key in ('pearson', 'spearman', 'kendall')})
    if len(gaps) and np.ptp(gaps) == 0:
        result.update(gap_r2=None, evaluation_mean_baseline_mse=0.)
    return result


def classification(y, scores):
    y, scores = np.asarray(y, float), np.asarray(scores, float)
    valid = np.isfinite(y) & np.isfinite(scores)
    y, scores = y[valid].astype(bool), scores[valid]
    return {'n': len(y), 'positive': int(y.sum()), 'auroc': auroc(y, scores),
            'average_precision': float(average_precision_score(y, scores)) if y.any() else None,
            'point_biserial': safe_corr(y, scores) if len(y) > 1 and y.any() and not y.all() and np.ptp(scores) > 0 else None}


def question_bootstrap(ids, values, *, samples, seed, tail_specs=None):
    """Resample questions and recompute reward quantiles in every bootstrap draw.

    Final evaluations have one answer per question; repeated-answer groups are
    also supported without treating their answers as independent questions.
    """
    if type(samples) is not int or samples < 0:
        raise ValueError('Bootstrap samples must be a nonnegative integer.')
    ids = np.asarray(ids)
    arrays = {k: np.asarray(v, float) for k, v in values.items()}
    if any(v.ndim != 1 or len(v) != len(ids) for v in arrays.values()):
        raise ValueError('Bootstrap values must align with question IDs.')
    tail_specs = tail_specs or {}
    groups = [np.flatnonzero(ids == q) for q in sorted(set(ids.tolist()))]
    draws = {k: [] for k in [*arrays, *tail_specs]}
    rng = np.random.default_rng(seed)
    def means(a, mask=None):
        good = np.isfinite(a) if mask is None else np.isfinite(a) & mask
        count = good.sum(axis=1)
        return np.divide(np.where(good, a, 0.).sum(axis=1), count,
                         out=np.full(len(a), np.nan), where=count > 0)
    for start in range(0, samples if groups else 0, 64):
        count = min(64, samples-start)
        selections = rng.integers(0, len(groups), size=(count, len(groups)))
        if all(len(g) == 1 for g in groups):
            index = np.asarray(groups).ravel()[selections]
        else:
            expanded = [np.concatenate([groups[j] for j in draw]) for draw in selections]
            width = max(map(len, expanded))
            index = np.full((count, width), -1, dtype=int)
            for i, draw in enumerate(expanded):
                index[i, :len(draw)] = draw
        selected_arrays = {k: np.where(index >= 0, a[index], np.nan) for k, a in arrays.items()}
        for name, selected in selected_arrays.items():
            draws[name].extend(means(selected).tolist())
        for name, (value_key, score_key, fraction, upper) in tail_specs.items():
            scores = selected_arrays[score_key]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)  # All-missing resamples remain NaN.
                cutoff = np.nanquantile(np.where(np.isfinite(scores), scores, np.nan),
                                        1-fraction if upper else fraction, axis=1, method='linear')
            mask = np.isfinite(scores) & (scores >= cutoff[:, None] if upper else scores <= cutoff[:, None])
            draws[name].extend(means(selected_arrays[value_key], mask).tolist())
    result = {}
    for name, values in draws.items():
        finite = np.asarray(values, float)
        finite = finite[np.isfinite(finite)]
        result[name] = {'ci95': np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                        'valid_resamples': len(finite), 'requested_resamples': samples,
                        'unit': 'question', 'tail_cutoffs_recomputed': name in tail_specs}
    return result


def distance_buckets(distances):
    distances = np.asarray(distances, float)
    valid = np.isfinite(distances)
    buckets = np.full(len(distances), -1, int)
    edges = np.quantile(distances[valid], [.2, .4, .6, .8]) if valid.any() else np.full(4, np.nan)
    # Keep all equal distances together, even when this leaves empty buckets.
    buckets[valid] = np.searchsorted(edges, distances[valid], side='left')
    return buckets, edges


def peak_to_final(trajectory):
    """Compare peaks with the last *monitor* checkpoint, never with test accuracy."""
    groups = defaultdict(list)
    for row in trajectory:
        if row['cohort'] == 'monitor':
            groups[row['seed'], row['arm'], row['estimator']].append(row)
    result = []
    metrics = ('strict_accuracy', 'numeric_accuracy', 'mean_proxy', 'mean_judge',
               'mean_true_gap', 'mean_corrected_reward', 'mean_optimized_reward', 'optimism_mean')
    for (seed, arm, estimator), rows in groups.items():
        if arm == 'base':
            continue
        rows = sorted(rows, key=lambda r: r['step'])
        initial = groups.get((seed, 'base', estimator), [])
        objective = 'knn_static' if arm == 'knn_refresh' else arm
        rows = [{**r, 'mean_optimized_reward': r.get('mean_'+objective+'_objective')} for r in initial] + rows
        final = rows[-1]
        for metric in metrics:
            eligible = [r for r in rows if r.get(metric) is not None]
            if not eligible or final.get(metric) is None:
                continue
            peak = max(eligible, key=lambda r: (r[metric], -r['step']))
            record = {'seed': seed, 'arm': arm, 'estimator': estimator, 'metric': metric,
                      'cohort': 'monitor', 'peak_rule': 'maximum; not necessarily desirable for gap or optimism',
                      'peak_step': peak['step'], 'last_monitor_step': final['step'],
                      'peak_value': peak[metric], 'last_monitor_value': final[metric],
                      'difference': final[metric]-peak[metric], 'available_checkpoints': len(rows)}
            for other in ('strict_accuracy', 'mean_judge', 'mean_optimized_reward', 'optimism_mean'):
                record[other+'_change_since_peak'] = (final[other]-peak[other]
                    if final.get(other) is not None and peak.get(other) is not None else None)
            result.append(record)
    return result


def cohorts(output):
    """Every saved checkpoint, not just the final or most recent monitor."""
    selection = output/'prepared/selection_scored.jsonl'
    if selection.exists():
        yield 'selection', 'base', 0, selection, output/'prepared/selection_features.npz'
    for kind in ('monitor', 'final'):
        for folder in sorted((output/'evaluations'/kind).glob('*/step_*')):
            if (folder/'metrics.json').exists():
                yield kind, folder.parent.name, int(folder.name.removeprefix('step_')), folder/'responses.jsonl', folder/'features.npz'


def frozen_definition(output):
    """Read an existing preselected target; analysis never selects a new cutoff."""
    for version in ('gap_validation_v3', 'gap_validation_v2'):
        path = output/'validation'/version/'judge_selection.json'
        if path.exists():
            saved = read_json(path)
            for relative, expected in saved.get('identity', {}).get('input_sha256', {}).items():
                if file_sha(output/relative) != expected:
                    raise ValueError('Frozen validation inputs changed; refusing a mismatched high-gap definition.')
            return saved['label_definition'], path
    return None, None


def _cohort_diagnostics(meta, rows, gaps, predictions, rewards, judge_z, distance, definition, tables):
    """One fixed set of answers for every estimator and reward signal."""
    ids, length = np.asarray([r['id'] for r in rows]), column(rows, 'response_tokens')
    valid = np.isfinite(gaps) & np.isfinite(predictions)
    residual = gaps-predictions
    high = label(gaps[valid], definition) if definition else None
    metrics = {**meta, 'answers': len(rows), 'scored_pairs': int(valid.sum()),
               **gap_statistics(gaps[valid], predictions[valid]),
               **optimism_statistics(residual), **optimistic_tail_bias(gaps, predictions, ids),
               **high_reward_tail_bias(residual, rewards, ids),
               'high_gap_auroc': auroc(high, predictions[valid]) if high is not None else None,
               'high_gap_average_precision': float(average_precision_score(high, predictions[valid])) if high is not None and high.any() else None,
               'high_gap_rate': float(high.mean()) if high is not None and len(high) else None,
               'mean_gap_when_high': mean(gaps[valid][high]) if high is not None else None}
    tables['gap_prediction'].append(metrics)
    buckets, edges = distance_buckets(distance)
    for bucket in range(5):
        selected = buckets == bucket
        scored = selected & valid
        y = label(gaps[scored], definition) if definition else None
        tables['distribution_shift'].append({**meta, 'distance_quintile': bucket+1,
            'answers': int(selected.sum()), 'scored_pairs': int(scored.sum()),
            'distance_lower_cutoff': number(edges[bucket-1]) if bucket else None,
            'distance_upper_cutoff': number(edges[bucket]) if bucket < 4 else None,
            'mean_nn_distance': mean(distance[selected]),
            **gap_statistics(gaps[scored], predictions[scored]), **optimism_statistics(residual[scored]),
            'high_gap_auroc': auroc(y, predictions[scored]) if y is not None else None})
    for name, values in (('predicted_gap', predictions), ('corrected_reward', rewards), ('actual_gap', gaps), ('judge_z', judge_z)):
        paired = np.isfinite(length) & np.isfinite(values)
        x, y = length[paired], values[paired]
        slope = float(np.mean((x-x.mean())*(y-y.mean()))/np.var(x)) if len(x) > 1 and np.var(x) > 0 else None
        tables['length_shortcuts'].append({**meta, 'target': name, **ranking(length, values),
            'slope_per_token': slope, 'intercept': float(y.mean()-slope*x.mean()) if slope is not None else None})
    return metrics


def _reward_diagnostics(meta, rows, signals, judge_z, tables):
    # All reward signals are compared on exactly the same finite-reward answers.
    common = np.isfinite(judge_z)
    for signal in signals.values():
        common &= np.isfinite(signal)
    ids = np.asarray([r['id'] for r in rows])
    for name, raw in signals.items():
        scores = np.where(common, raw, np.nan)
        aligned = ranking(scores, judge_z)
        for target in ('correct', 'numeric_match'):
            labels = column(rows, target)
            tables['reward_correctness_alignment'].append({**meta, 'reward_signal': name, 'correctness': target,
                'all_answers': len(rows), 'common_reward_answers': int(common.sum()),
                **classification(labels, scores), **{'judge_'+k: v for k, v in aligned.items()}})
            for fraction in (*TOP_FRACTIONS, .5, 1.):
                selected, cutoff = top_tail(scores, fraction)
                valid = selected & np.isfinite(labels)
                tables['top_reward_accuracy'].append({**meta, 'reward_signal': name, 'correctness': target,
                    'top_fraction': fraction, 'reward_cutoff': cutoff,
                    'common_reward_answers': int(common.sum()), 'selected_answers': int(selected.sum()),
                    'selected_fraction': float(selected.sum()/common.sum()) if common.any() else None,
                    'selected_questions': len(set(ids[selected].tolist())),
                    'labeled_answers': int(valid.sum()), 'accuracy': mean(labels[valid])})


def _analyze_seed(output, tables, statistics, stream, audit, bootstrap_samples):
    config = read_json(output/'config.json')
    seed = config['seed']
    bank, norm_path = output/'prepared/memory_matched.npz', output/'prepared/normalization.json'
    if not bank.exists() or not norm_path.exists():
        audit['unavailable'].append({'seed': seed, 'reason': 'matched memory or normalization is unavailable'})
        return
    norm, memory = Normalization.load(norm_path), GapMemory.load(bank)
    ridge_folder = output/'prepared/ridge'
    ridge = RidgeGap.load(ridge_folder, memory) if (ridge_folder/'selection.json').exists() else None
    if ridge is None:
        audit['unavailable'].append({'seed': seed, 'reason': 'ridge coefficients are unavailable; reporting existing predictors'})
    definition, locked_path = frozen_definition(output)
    inputs = [output/'config.json', bank, norm_path]
    if ridge is not None:
        inputs += [ridge_folder/'selection.json', ridge_folder/'ridge.npz']
    if locked_path:
        inputs.append(locked_path)
    else:
        audit['unavailable'].append({'seed': seed, 'reason': 'no frozen high-gap target; classification metrics unavailable'})
    strong_path = output/'prepared_30b/memory_initial.npz'
    strong_norm_path = output/'prepared_30b/normalization.json'
    strong = GapMemory.load(strong_path) if strong_path.exists() and strong_norm_path.exists() else None
    strong_norm = Normalization.load(strong_norm_path) if strong is not None else None
    if strong is not None:
        inputs += [strong_path, strong_norm_path]
    samples = config['evaluation']['bootstrap_samples'] if bootstrap_samples is None else bootstrap_samples
    mode = config['knn']['correction']
    seen, final_answers = {}, {}
    for kind, arm, step, path, features in cohorts(output):
        if not features.exists():
            audit['unavailable'].append({'seed': seed, 'cohort': kind, 'arm': arm, 'step': step,
                                          'reason': 'saved features missing; historical answers were not regenerated'})
            continue
        rows = read_jsonl(path)
        if not rows:
            continue
        ids = np.asarray([r['id'] for r in rows])
        if set(ids) & set(memory.group_ids):
            raise ValueError('Analysis evaluation questions overlap fitting memory.')
        if kind != 'selection':
            questions = {r['id']: (r['question'], r['reference']) for r in rows}
            if len(questions) != len(rows) or (kind in seen and seen[kind] != questions):
                raise ValueError('Checkpoint evaluations need one answer per identical question/reference.')
            seen[kind] = questions
        inputs += [path, features]
        print(f'Analyze seed {seed}: {kind}/{arm}/{step} ({len(rows)} answers)', flush=True)
        x = load_features(features, rows, memory.encoder_identity)
        proxy_raw, judge_raw = column(rows, 'proxy_score'), column(rows, 'judge_score')
        zp, zj = norm.proxy_z(proxy_raw), norm.judge_z(judge_raw)
        gaps = zp-zj
        knn, similarity, neighbors = memory.predict(x, ids, memory.encoder_identity)
        distance = 1-similarity
        predictions = {'mean_gap': np.full(len(rows), float(memory.gaps.mean())), 'knn_static': knn}
        if ridge is not None:
            predictions['ridge'] = ridge.predict(x, ids, memory.encoder_identity)[0]
        rewards = {name: corrected_reward(zp, pred, mode) for name, pred in predictions.items()}
        signals = {'proxy': zp, 'judge': zj, **rewards}
        penalty_config = config.get('completion_reward', {})
        penalty = (penalty_config.get('format_penalty', 0.)*(1-column(rows, 'format_valid')) +
                   penalty_config.get('incomplete_penalty', 0.)*np.asarray([
                       r.get('length_capped', False) or not r.get('ended_with_eos', True) for r in rows]))
        objectives = {name: reward-penalty for name, reward in signals.items()}
        objectives['oracle'] = column(rows, 'correct')-penalty
        if strong is not None and arm in ('base', 'knn_static_30b'):
            strong_prediction = strong.predict(x, ids, memory.encoder_identity)[0]
            objectives['knn_static_30b'] = corrected_reward(strong_norm.proxy_z(proxy_raw), strong_prediction, mode)-penalty
        if arm == 'knn_refresh':
            choices = sorted(p for p in (output/'arms/knn_refresh/memories').glob('step_*.npz') if int(p.stem.split('_')[-1]) <= step)
            active = GapMemory.load(choices[-1]) if choices else memory
            inputs += choices[-1:]
            objectives[arm] = corrected_reward(zp, active.predict(x, ids, memory.encoder_identity)[0], mode)-penalty
        meta = {'seed': seed, 'cohort': kind, 'arm': arm, 'step': step,
                'selection_diagnostic': kind == 'selection', 'correction_mode': mode}
        _reward_diagnostics(meta, rows, signals, zj, tables)
        training_path = output/'arms'/arm/'training'/f'step_{step:06d}.json'
        training = read_json(training_path) if training_path.exists() else {}
        if training_path.exists():
            inputs.append(training_path)
        policy_metrics = {**meta, 'answers': len(rows), 'paired_labels': int(np.isfinite(gaps).sum()),
            'strict_accuracy': mean(column(rows, 'correct')), 'numeric_accuracy': mean(column(rows, 'numeric_match')),
            'mean_proxy': mean(proxy_raw), 'mean_judge': mean(judge_raw),
            'mean_proxy_z': mean(zp), 'mean_judge_z': mean(zj), 'mean_true_gap': mean(gaps),
            'mean_length': mean(column(rows, 'response_tokens')), 'valid_format_rate': mean(column(rows, 'format_valid')),
            'eos_rate': mean(column(rows, 'ended_with_eos')), 'length_cap_rate': mean(column(rows, 'length_capped')),
            'mean_nn_similarity': mean(similarity), 'mean_nn_distance': mean(distance),
            'mean_optimized_reward': mean(objectives[arm]) if arm in objectives else None,
            **{'mean_'+name+'_objective': mean(value) for name, value in objectives.items()},
            'sampled_training_reference_kl': training.get('sampled_reference_kl_per_response'),
            'training_kl_scope': 'sampled training rollout at this attempt; not evaluation KL',
            'source_responses': str(path), 'source_features': str(features)}
        if kind == 'final':
            tables['ppo_final_results'].append(policy_metrics)
            final_answers[arm] = rows
        for estimator, pred in predictions.items():
            reward = rewards[estimator]
            metrics = _cohort_diagnostics({**meta, 'estimator': estimator}, rows, gaps, pred, reward, zj, distance, definition, tables)
            checkpoint = {**policy_metrics, **metrics, 'mean_pred_gap': mean(pred),
                          'mean_corrected_reward': mean(reward), 'mean_applied_reward_error': mean(reward-zj)}
            tables['ppo_checkpoint_metrics'].append(checkpoint)
            if kind == 'final':
                values = {'strict_accuracy': column(rows, 'correct'), 'numeric_accuracy': column(rows, 'numeric_match'),
                    'mean_proxy': proxy_raw, 'mean_judge': judge_raw, 'mean_true_gap': gaps,
                    'optimism_mean': gaps-pred, 'mean_corrected_reward': reward, 'mean_pred_gap': pred}
                tails = {f'high_reward_tail_optimism_{round(q*100):02d}': ('optimism_mean', 'mean_corrected_reward', q, True) for q in TOP_FRACTIONS}
                tails |= {f'optimistic_tail_bias_{round(q*100):02d}': ('optimism_mean', 'mean_pred_gap', q, False) for q in (.01, .05, .1)}
                statistics['bootstrap_cis'].append({**meta, 'estimator': estimator,
                    'metrics': question_bootstrap(ids, values, samples=samples,
                        seed=seed_for(seed, kind, arm, 'analysis-bootstrap'), tail_specs=tails)})
            for i, row in enumerate(rows):
                record = {**meta, 'estimator': estimator, 'prompt_id': row['id'], 'prompt': row['question'],
                    'response': row['response'], 'reference': row['reference'],
                    'strict_correct': row.get('correct'), 'numeric_correct': row.get('numeric_match'),
                    'valid_format': row.get('format_valid'), 'num_tokens': row.get('response_tokens'),
                    'proxy_raw': number(proxy_raw[i]), 'judge_raw': number(judge_raw[i]),
                    'z_proxy': number(zp[i]), 'z_judge': number(zj[i]), 'true_gap': number(gaps[i]),
                    'pred_gap': number(pred[i]), 'corrected_reward': number(reward[i]),
                    'optimism': number(gaps[i]-pred[i]), 'applied_reward_error': number(reward[i]-zj[i]),
                    'nn_similarity_1': number(similarity[i]), 'nn_distance_1': number(distance[i]),
                    'feature_file': str(features), 'feature_row': i,
                    'ridge_projection': number(pred[i]-ridge.intercept) if estimator == 'ridge' else None,
                    'knn_neighbor_ids': memory.group_ids[neighbors[i]].tolist() if estimator == 'knn_static' else None,
                    'knn_neighbor_gaps': memory.gaps[neighbors[i]].tolist() if estimator == 'knn_static' else None}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
    _paired_final(seed, final_answers, samples, statistics)
    audit['seeds'].append({'seed': seed, 'correction_mode': mode, 'frozen_high_gap_definition': definition,
                           'memory_answers': len(memory.gaps), 'mean_baseline': float(memory.gaps.mean()),
                           'bootstrap_samples': samples})
    audit['inputs'].update({str(path): file_sha(path) for path in set(inputs)})


def _paired_final(seed, final_answers, samples, statistics):
    arms = sorted(final_answers)
    for i, left in enumerate(arms):
        for right in arms[i+1:]:
            lhs = {r['id']: r for r in final_answers[left]}
            rhs = {r['id']: r for r in final_answers[right]}
            if set(lhs) != set(rhs):
                raise ValueError('Paired final tests require identical question IDs.')
            for target in ('correct', 'numeric_match'):
                ids = sorted(k for k in lhs if type(lhs[k].get(target)) is bool and type(rhs[k].get(target)) is bool)
                a, b = np.asarray([lhs[k][target] for k in ids], bool), np.asarray([rhs[k][target] for k in ids], bool)
                win, lose = int((a & ~b).sum()), int((~a & b).sum())
                difference = a.astype(float)-b.astype(float)
                ci = question_bootstrap(ids, {'difference': difference}, samples=samples,
                    seed=seed_for(seed, left, right, target, 'paired'))['difference']
                statistics['significance_tests'].append({'seed': seed, 'left': left, 'right': right,
                    'correctness': target, 'questions': len(ids), 'excluded_unlabeled_pairs': len(lhs)-len(ids), 'difference': mean(difference),
                    'paired_bootstrap': ci, 'left_only_correct': win, 'right_only_correct': lose,
                    'mcnemar_exact_p': (float(binomtest(win, win+lose, .5).pvalue) if win+lose else 1.) if ids else None,
                    'multiplicity': 'unadjusted exploratory pairwise p-values; no automatic significance claims'})


def _robustness_table(tables):
    trajectory = tables['ppo_checkpoint_metrics']
    base = {(r['seed'], r['estimator']): r for r in trajectory if r['cohort'] == 'final' and r['arm'] == 'base'}
    for row in trajectory:
        if row['cohort'] != 'final' or row['arm'] == 'base':
            continue
        offline = base.get((row['seed'], row['estimator']), {})
        monitor = [r for r in trajectory if r['cohort'] == 'monitor' and r['seed'] == row['seed']
                   and r['estimator'] == row['estimator'] and r['arm'] in ('base', row['arm'])]
        last = max(monitor, key=lambda r: r['step']) if monitor else {}
        eligible = [r for r in monitor if r.get('strict_accuracy') is not None]
        best = max(eligible, key=lambda r: (r['strict_accuracy'], -r['step'])) if eligible else {}
        top = next((r for r in tables['top_reward_accuracy'] if r['seed'] == row['seed'] and
            r['cohort'] == 'final' and r['arm'] == row['arm'] and r['reward_signal'] == row['estimator']
            and r['correctness'] == 'correct' and r['top_fraction'] == .05), {})
        tables['optimization_robustness'].append({
            'seed': row['seed'], 'ppo_arm': row['arm'], 'estimator': row['estimator'],
            'offline_source': 'final/base (identical initial-policy test answers)',
            'offline_mse': offline.get('gap_mse'), 'ppo_distribution_mse': row['gap_mse'],
            'mean_optimism': row['optimism_mean'], 'p95_optimism': row['optimism_p95'],
            'top_5pct_reward_optimism': row['high_reward_tail_optimism_05'],
            'top_5pct_reward_correctness': top.get('accuracy'),
            'top_5pct_reward_answers': top.get('selected_answers'),
            'peak_monitor_step': best.get('step'), 'peak_monitor_accuracy': best.get('strict_accuracy'),
            'last_monitor_step': last.get('step'), 'last_monitor_accuracy': last.get('strict_accuracy'),
            'monitor_peak_to_last': last['strict_accuracy']-best['strict_accuracy'] if best else None,
            'final_test_accuracy': row['strict_accuracy'], 'final_test_questions': row['answers']})


def _seed_summary(tables):
    rows = []
    for table, keys in (('gap_prediction', ('cohort', 'arm', 'step', 'estimator')),
                        ('ppo_final_results', ('arm',)),
                        ('optimization_robustness', ('ppo_arm', 'estimator'))):
        grouped = defaultdict(list)
        for record in tables[table]:
            grouped[tuple(record[k] for k in keys)].append(record)
        for key, records in sorted(grouped.items()):
            if len({r['seed'] for r in records}) != len(records):
                raise ValueError('Duplicate seed/cohort records in analysis summary.')
            numeric = {k for r in records for k, v in r.items() if type(v) in (int, float) and k not in (*keys, 'seed')}
            for metric in sorted(numeric):
                values = [r[metric] for r in records if r.get(metric) is not None]
                rows.append({'table': table, **dict(zip(keys, key)), 'metric': metric,
                    'n_seeds': len(values), 'seeds_available': len(records),
                    'mean': mean(values), 'sample_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None})
    return rows


def analyze(output, destination=None, *, bootstrap_samples=None, figures=True):
    """One command for a single seed directory or the root of a seed suite."""
    output = Path(output).resolve()
    destination = Path(destination).resolve() if destination else output/'paper_results'
    if not (output/'config.json').exists() and not (output/'suite_protocol.json').exists():
        raise ValueError('Pass a standalone experiment output or seed-suite output directory.')
    if destination == output or destination in output.parents:
        raise ValueError('Choose a separate analysis destination, not the source or its parent.')
    if destination.is_relative_to(output) and not destination.is_relative_to(output/'paper_results'):
        raise ValueError('Within a run, analysis outputs must be under paper_results/.')
    if bootstrap_samples is not None and (type(bootstrap_samples) is not int or bootstrap_samples < 0):
        raise ValueError('--bootstrap-samples must be a nonnegative integer.')
    if (output/'suite_protocol.json').exists():
        seeds = read_json(output/'suite_protocol.json')['seeds']
        sources = [output/f'seed_{seed}' for seed in seeds]
    else:
        sources = [output]
    # A marker distinguishes this analysis output from arbitrary existing data.
    marker = destination/'analysis.json'
    if destination.exists() and any(destination.iterdir()) and not marker.exists():
        raise ValueError('Analysis destination is not empty and has no analysis.json marker.')
    if marker.exists() and read_json(marker).get('source') != str(output):
        raise ValueError('Analysis destination belongs to a different experiment.')
    destination.mkdir(parents=True, exist_ok=True)
    audit = {'version': VERSION, 'source': str(output), 'status': 'running', 'seeds': [], 'inputs': {}, 'unavailable': [],
        'models_refitted': False, 'new_judge_calls': 0, 'final_used_for_tuning': False,
        'method': {'optimism': 'actual normalized gap minus predicted gap',
            'reward_tail': 'top 1/5/10/20 percent of corrected reward, inclusive linear quantiles',
            'gap_tail': 'bottom 1/5/10 percent of predicted gap; retained as a separate metric',
            'mean_baseline': 'mean actual gap in the same matched fitting memory; no evaluation labels',
            'reward_comparisons': 'all signals use the same finite-reward answers within each cohort',
            'distance_buckets': 'quintile cutoffs per cohort; ties stay together, so buckets can be empty',
            'embedding': 'frozen proxy grader representation; changing answers, not changing encoder weights',
            'optimized_reward': 'arm terminal reward including completion penalties, before token-level KL; sampled training KL is logged separately',
            'bootstrap': 'resample whole questions; recompute tail quantiles; conditional on fitted models and available labels',
            'seeds': 'per-seed metrics then mean and sample SD; no pooling questions across seeds'},
        'analysis_source_sha256': {p.name: file_sha(p) for p in [Path(__file__), Path(__file__).with_name('metrics.py'),
            Path(__file__).with_name('memory.py'), Path(__file__).with_name('validation.py'), Path(__file__).with_name('analysis_plots.py')]}}
    atomic_json(marker, audit)
    tables = defaultdict(list)
    statistics = {'bootstrap_cis': [], 'significance_tests': []}
    (destination/'raw').mkdir(exist_ok=True)
    try:
        with gzip.open(destination/'raw/sample_level_metrics.jsonl.gz', 'wt', encoding='utf-8') as stream:
            with threadpool_limits(limits=8):
                for source in sources:
                    if not (source/'config.json').exists():
                        audit['unavailable'].append({'source': str(source), 'reason': 'seed has not started'})
                        continue
                    _analyze_seed(source, tables, statistics, stream, audit, bootstrap_samples)
        tables['peak_to_final'] = peak_to_final(tables['ppo_checkpoint_metrics'])
        _robustness_table(tables)
        tables['seed_summary'] = _seed_summary(tables)
        for name, rows in tables.items():
            folder = 'raw' if name == 'ppo_checkpoint_metrics' else 'tables'
            write_csv(destination/folder/f'{name}.csv', rows)
        for name, rows in statistics.items():
            atomic_json(destination/'stats'/f'{name}.json', {'scope': audit['method']['bootstrap'], 'records': rows})
        if figures:
            from .analysis_plots import make_figures
            make_figures(tables, destination/'figures')
        audit.update(status='complete', table_rows={k: len(v) for k, v in tables.items()}, figures_generated=figures)
        _write_report(tables, statistics, audit, destination)
        atomic_json(marker, audit)
    except Exception as error:
        atomic_json(marker, {**audit, 'status': 'failed', 'error': str(error)})
        raise
    print(f'Paper analysis: {destination / "report.md"}', flush=True)
    return destination


def _write_report(tables, statistics, audit, destination):
    def fmt(v):
        return 'unavailable' if v is None else f'{v:.4f}'
    def table(headers, rows):
        return ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                *['| '+' | '.join(map(str, row))+' |' for row in rows], '']
    lines = ['# GSM8K prediction versus optimization', '',
        f"Analysis `{VERSION}`. No generation, model refitting, or new judge calls. Seeds analyzed: {[s['seed'] for s in audit['seeds']]}", '',
        'This report tests whether offline prediction quality agrees with optimization behavior. It does not assume '
        'ridge is exploited or kNN is safer. A judge remains an imperfect evaluator; numeric task correctness '
        'is an independent check of final answers, not of every reasoning step.', '',
        '## Gap prediction on shared base-policy test answers', '']
    static = [r for r in tables['gap_prediction'] if r['cohort'] == 'final' and r['arm'] == 'base']
    lines += table(['Seed', 'Predictor', 'Pairs', 'MSE', 'R2', 'Kendall', 'AUROC', 'AP', 'Mean optimism', 'Top-5%-reward optimism'],
        [[r['seed'], r['estimator'], r['scored_pairs'], *[fmt(r[k]) for k in ('gap_mse', 'gap_r2', 'gap_kendall',
          'high_gap_auroc', 'high_gap_average_precision', 'optimism_mean', 'high_reward_tail_optimism_05')]] for r in static])
    lines += ['## Final policies', '']
    ci = {(r['seed'], r['arm']): r['metrics']['strict_accuracy']['ci95'] for r in statistics['bootstrap_cis']}
    lines += table(['Seed', 'Policy', 'Strict accuracy', '95% question-bootstrap CI', 'Numeric accuracy', 'Judge', 'Actual gap', 'Tokens', 'EOS rate'],
        [[r['seed'], r['arm'], fmt(r['strict_accuracy']), str(ci.get((r['seed'], r['arm']))),
          *[fmt(r[k]) for k in ('numeric_accuracy', 'mean_judge', 'mean_true_gap', 'mean_length', 'eos_rate')]] for r in tables['ppo_final_results']])
    lines += ['## Optimization robustness', '',
        'Offline MSE uses identical base-policy final answers. PPO-distribution MSE uses each trained policy\'s final answers. '
        'Peak-to-last changes use only the shared monitoring cohort; test accuracy is a separate endpoint.', '']
    lines += table(['Seed', 'Method', 'Offline MSE', 'PPO MSE', 'Mean optimism', 'P95 optimism', 'Top-5% reward accuracy', 'Tail n', 'Monitor peak-to-last', 'Test accuracy'],
        [[r['seed'], r['estimator'], *[fmt(r[k]) for k in ('offline_mse', 'ppo_distribution_mse', 'mean_optimism',
          'p95_optimism', 'top_5pct_reward_correctness')], r['top_5pct_reward_answers'], fmt(r['monitor_peak_to_last']), fmt(r['final_test_accuracy'])]
         for r in tables['optimization_robustness'] if r['ppo_arm'] == r['estimator']])
    lines += ['## Reading the artifacts', '',
        '- `tables/gap_prediction.csv`: every estimator on every saved evaluation checkpoint, including mean-gap baseline, Kendall and both tail definitions.',
        '- `tables/top_reward_accuracy.csv`: strict/numeric correctness at top 1/5/10/20/50/100%, using identical finite-reward answers for all signals.',
        '- `tables/reward_correctness_alignment.csv`: correctness AUROC/AP and Pearson/Spearman/Kendall alignment with judge reward.',
        '- `raw/ppo_checkpoint_metrics.csv`: all saved monitor steps and final endpoints, scores, gaps, optimism, format, EOS, lengths and memory distance.',
        '- `tables/distribution_shift.csv`, `tables/length_shortcuts.csv`: distance-quintile errors and length correlations/regressions.',
        '- `tables/peak_to_final.csv`: per-metric maxima versus last monitor checkpoint, including changes in the optimized objective and judge quality from that same peak.',
        '- `stats/bootstrap_cis.json`: question-bootstrap intervals; tail cutoffs are recomputed in each resample.',
        '- `stats/significance_tests.json`: paired accuracy intervals and exact McNemar tests; p-values are exploratory and unadjusted.',
        '- `tables/seed_summary.csv`: means and sample standard deviations across seeds, with contributing seed counts.',
        '- `raw/sample_level_metrics.jsonl.gz`: individual answers, scores, gaps, rewards, optimism, neighbors and exact feature-file references.',
        ('- `figures/`: standalone PDF/PNG plots derived from the tables.' if audit['figures_generated'] else
         '- Figure generation disabled for this analysis; any preexisting figure files were not refreshed.'), '',
        '## Interpretation and coverage', '',
        'Positive gap optimism g - g_hat equals corrected reward minus normalized judge reward for signed correction. '
        'For positive_only correction, inspect applied_reward_error separately. High-reward tails condition on corrected reward, '
        'and are distinct from the retained low-predicted-gap tails. Ties can expand either tail. Missing grades stay missing; '
        'tail averages use available scored pairs and report support. Tiny tails can give unstable or degenerate bootstrap intervals.', '',
        'Checkpoint curves are monitoring diagnostics, not repeated tests on the official test set. Peak selection is descriptive; '
        'no claim of statistical significance is attached to a chosen peak. A decline in accuracy alone is not evidence of reward hacking. '
        'Check whether the optimized terminal reward improves while judge quality or correctness declines. '
        'mean_optimized_reward includes completion penalties but excludes token-level KL; the sampled training KL is reported separately.', '',
        'The mean baseline is fitted only on memory gaps. Selection rows are tuning diagnostics. This benchmark may already '
        'have been inspected. Bootstrap intervals condition on this fitted memory, normalization and predictor; they do not '
        'measure training-seed, calibration or missing-label uncertainty. AP denotes average precision, not trapezoidal PR area.', '',
        'This core analysis uses paired checkpoint evaluations. It does not add judge calls to every training rollout. '
        'Selection regret requires multiple held-out candidates per prompt; default monitor/final evaluations contain one. '
        'Representation-layer/weighting/memory-size ablations, causal ridge-direction tests, new training seeds and paper rewriting '
        'are outside this core addition. Existing optional refresh/30B training is not rerun. All predictors in the main gap tables '
        'target the common 4B judge; 30B PPO objective values, when available, use that arm\'s own normalization and memory.', '']
    if audit['unavailable']:
        lines += ['## Unavailable inputs', '', *[f'- {item}' for item in audit['unavailable']], '']
    (destination/'report.md').write_text('\n'.join(lines), encoding='utf-8')
