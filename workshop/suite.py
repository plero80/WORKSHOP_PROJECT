"""Sequential seeds on one GPU; each seed retains an isolated experiment directory."""
from __future__ import annotations

import copy
import statistics
import subprocess
import sys
from pathlib import Path

from .common import ROOT, atomic_json, read_json, run_lock, write_csv
from .report import LABELS
from .metrics import TAIL_BIAS_KEYS, TAIL_BIAS_PROTOCOL


def aggregate(output, seeds, arms):
    records, differences, predictor_rows = [], {}, []
    pairs = [(arm, 'proxy') for arm in arms if arm != 'proxy']
    pairs += [(arm, 'knn_static') for arm in ('ridge', 'knn_static_30b') if arm in arms]
    for seed in seeds:
        folder = Path(output) / f'seed_{seed}'
        marker = read_json(folder / 'final_protocol.json')
        summary = read_json(folder / 'summary.json')
        if marker['arms'] != arms:
            raise ValueError('Cannot aggregate different arm lists.')
        by_arm = {r['arm']: r for r in summary['metrics'] if r['cohort'] == 'final'}
        missing = {'base', *arms} - set(by_arm)
        if missing - set(summary.get('skipped_arms', {})) or set(by_arm) - {'base', *arms}:
            raise ValueError('A seed is missing final evaluations.')
        records.append({'seed': seed, 'metrics': by_arm, 'skipped_arms': summary.get('skipped_arms', {})})
        for left, right in pairs:
            if {left, right} <= set(by_arm):
                differences.setdefault(f'{left}_minus_{right}', []).append({'seed': seed,
                    'strict_difference': by_arm[left]['accuracy'] - by_arm[right]['accuracy'],
                    'numeric_difference': by_arm[left]['numeric_accuracy'] - by_arm[right]['numeric_accuracy']})
        predictions = folder/'predictors/summary.json'
        if predictions.exists():
            predictor_rows.extend(r for r in read_json(predictions)['metrics'] if r['cohort'].startswith('final/'))
    def stats(values):
        values = [v for v in values if v is not None]
        return {'mean': statistics.mean(values) if values else None, 'sample_sd': statistics.stdev(values) if len(values) > 1 else None, 'values': values, 'n_seeds': len(values)}
    summary = {arm: {key: stats([r['metrics'][arm].get(key) for r in records if arm in r['metrics']])
                    for key in ('accuracy', 'numeric_accuracy', 'numeric_unresolved_rate', 'format_valid_rate', 'length_cap_rate',
                                'mean_judge_score', 'high_gap_rate', 'mean_response_tokens')}
               for arm in ['base', *arms]}
    paired = {pair: {key: stats([r[key] for r in rows]) for key in ('strict_difference', 'numeric_difference')}
              for pair, rows in differences.items()}
    predictor_summary = []
    for cohort, predictor in sorted({(r['cohort'], r['predictor']) for r in predictor_rows}):
        selected = [r for r in predictor_rows if (r['cohort'], r['predictor']) == (cohort, predictor)]
        for key in ('gap_mse', 'gap_rmse', 'gap_mae', 'gap_r2', 'gap_pearson', 'gap_spearman', 'high_gap_auroc', 'high_gap_average_precision',
                    'answer_policy_accuracy', 'answer_policy_numeric_accuracy', *TAIL_BIAS_KEYS,
                    *(k+suffix for k in TAIL_BIAS_KEYS for suffix in ('_n', '_selected', '_questions', '_unscored', '_fraction', '_cutoff'))):
            predictor_summary.append({'cohort': cohort, 'predictor': predictor, 'metric': key,
                                      **stats([r.get(key) for r in selected])})
    write_csv(Path(output)/'suite_predictors.csv', predictor_summary)
    atomic_json(Path(output) / 'suite_summary.json', {'seeds': seeds, 'arms': summary, 'per_seed': records,
        'paired_differences': paired, 'per_seed_differences': differences,
        'optimistic_tail_bias_protocol': TAIL_BIAS_PROTOCOL,
        'predictors': predictor_summary, 'per_seed_predictors': predictor_rows})
    lines = ['# GSM8K matched-seed suite', '', f'Seeds: {seeds}. Values below are mean ± sample standard deviation across training seeds, not confidence intervals.', '',
             '| Policy | Strict accuracy | Numeric matches / all | Valid box | Unresolved |', '|---|---:|---:|---:|---:|']
    def fmt(v):
        if v['mean'] is None:
            return 'unavailable (no graded training data)'
        return f"{v['mean']:.2%} ± {v['sample_sd']:.2%}" if v['sample_sd'] is not None else f"{v['mean']:.2%} (one seed)"
    for arm, values in summary.items():
        lines.append(f"| {LABELS.get(arm, arm)} | {fmt(values['accuracy'])} | {fmt(values['numeric_accuracy'])} | {fmt(values['format_valid_rate'])} | {fmt(values['numeric_unresolved_rate'])} |")
    base_predictors = {(r['predictor'], r['metric']): r for r in predictor_summary if r['cohort'] == 'final/base/0'}
    if base_predictors:
        lines += ['', 'Predictor diagnostics on identical base-policy final answers, alongside each method\'s PPO accuracy. '
                  'Entries are means +/- sample standard deviations across seeds, not confidence intervals. '
                  'n after OTB is the mean scored tail count; per-seed counts and missing labels are in the JSON/CSV.', '',
                  '| Method | MSE | R2 | AUROC | OTB 1% (n) | OTB 5% (n) | OTB 10% (n) | PPO strict accuracy |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        def diagnostic(method, metric):
            value = base_predictors.get((method, metric), {})
            if value.get('mean') is None:
                return 'unavailable'
            spread = f" +/- {value['sample_sd']:.4f}" if value['sample_sd'] is not None else ' (one seed)'
            return f"{value['mean']:.4f}"+spread
        for method in sorted({m for m, _ in base_predictors}):
            cells = [method, *[diagnostic(method, k) for k in ('gap_mse', 'gap_r2', 'high_gap_auroc')]]
            for key in TAIL_BIAS_KEYS:
                n = base_predictors.get((method, key+'_n'), {}).get('mean')
                cells.append(f"{diagnostic(method, key)} (n={n:g})" if n is not None else diagnostic(method, key))
            cells.append(fmt(summary[method]['accuracy']) if method in summary else 'unavailable')
            lines.append('| '+' | '.join(cells)+' |')
    lines += ['', 'Within-seed policy differences:', '']
    for pair, values in paired.items():
        for key, value in values.items():
            lines.append(f"- {pair}, {key}: {fmt(value)}; per-seed values {value['values']}.")
    lines += ['', 'Every seed has its own `report.md` and `predictors/report.md`. '
              '`suite_summary.json` includes every seed, mean judge scores, high-gap rates, response lengths, '
              'and predictor metrics. `suite_predictors.csv` gives their means and sample standard deviations. '
              'Unavailable metrics are excluded with the contributing seed count reported.', '']
    lines += ['', 'The data partition is held fixed across seeds. Policy initialization, sampled candidates, and PPO randomness vary by seed. '
              'Each seed uses matched memory responses across its two teachers. Three seeds provide limited evidence about variability; inspect every seed.', '']
    (Path(output) / 'suite_report.md').write_text('\n'.join(lines), encoding='utf-8')


def run_suite(config, output, seeds, stage='full', updates=None):
    """Fresh process per seed, one training runner and one reproducible configuration."""
    output = Path(output).resolve()
    with run_lock(output):
        declaration = {'seeds': seeds, 'config': config}
        path = output / 'suite_protocol.json'
        if path.exists():
            previous = read_json(path)
            old_seeds = previous['seeds']
            if previous['config'] != config or seeds[:len(old_seeds)] != old_seeds:
                raise ValueError('Suite configuration or existing seed order changed. Use another output directory.')
            if old_seeds != seeds:
                from .common import append_jsonl
                append_jsonl(output/'seed_additions.jsonl', {'previous': old_seeds, 'added': seeds[len(old_seeds):],
                    'note': 'Additional seeds declared after the initial suite; inspect the per-seed results.'})
        atomic_json(path, declaration)
        first = output / f'seed_{seeds[0]}'
        for seed in seeds:
            current = copy.deepcopy(config)
            current['seed'] = seed
            current['data_seed'] = config.get('data_seed', config['seed'])
            config_path = output / 'configs' / f'seed_{seed}.json'
            atomic_json(config_path, current)
            destination = output / f'seed_{seed}'
            resolved = first / 'resolved_assets.json'
            if destination != first and resolved.exists():
                pinned = destination / 'resolved_assets.json'
                if pinned.exists() and read_json(pinned) != read_json(resolved):
                    raise ValueError('Seeds must use identical model revisions.')
                atomic_json(pinned, read_json(resolved))
            destination.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', '-m', 'workshop.run', '--config', str(config_path),
                       '--output', str(destination), '--stage', stage]
            if updates is not None:
                command += ['--updates', str(updates)]
            print(f'Start/resume seed {seed}: {destination}\nLog: {destination / "experiment.log"}', flush=True)
            atomic_json(output / 'suite_status.json', {'stage': 'running', 'seed': seed, 'log': str(destination / 'experiment.log')})
            with (destination / 'experiment.log').open('a', encoding='utf-8') as log:
                code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if code:
                atomic_json(output / 'suite_status.json', {'stage': 'failed', 'seed': seed, 'returncode': code})
                raise RuntimeError(f'Seed {seed} stopped. See {destination / "experiment.log"}; remaining seeds were not launched.')
        if stage == 'full':
            aggregate(output, seeds, config['arms'])
            atomic_json(output/'suite_status.json', {'stage': 'analysis', 'seeds': seeds, 'training_complete': True})
            from .analysis import analyze
            analyze(output)
        atomic_json(output / 'suite_status.json', {'stage': 'complete', 'run_stage': stage, 'seeds': seeds})
