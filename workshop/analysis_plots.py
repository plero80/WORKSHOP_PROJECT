"""Publication-exportable plots from the core analysis tables, with no model work."""
from collections import defaultdict


def make_figures(tables, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True, exist_ok=True)

    def finish(fig, name):
        for ax in fig.axes:
            ax.grid(alpha=.2)
            if ax.lines or ax.collections:
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(destination/(name+'.pdf'), bbox_inches='tight')
        fig.savefig(destination/(name+'.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    trajectory = tables['ppo_checkpoint_metrics']
    monitor = [r for r in trajectory if r['cohort'] == 'monitor']
    base = {(r['seed'], r['estimator']): r for r in monitor if r['arm'] == 'base'}
    owned = [r for r in monitor if r['arm'] == r['estimator'] and r['arm'] in ('ridge', 'knn_static')]

    def curves(rows, key, title, name, ylabel):
        fig, ax = plt.subplots(figsize=(8, 4))
        grouped = defaultdict(list)
        for r in rows:
            if r['arm'] != 'base':
                grouped[r['seed'], r['arm'], r['estimator']].append(r)
        for (seed, arm, estimator), values in sorted(grouped.items()):
            initial = base.get((seed, estimator))
            values = ([initial] if initial else [])+sorted(values, key=lambda r: r['step'])
            values = [r for r in values if r.get(key) is not None]
            if values:
                line, = ax.plot([r['step'] for r in values], [r[key] for r in values], marker='.', label=f'{arm}, seed {seed}')
                if key == 'strict_accuracy':
                    best = max(values, key=lambda r: r[key])
                    ax.scatter([best['step']], [best[key]], marker='*', s=85, color=line.get_color(), zorder=3)
        ax.set(title=title, xlabel='PPO attempt (monitor cohort)', ylabel=ylabel)
        if key == 'strict_accuracy':
            ax.set_ylim(0, 1)
        finish(fig, name)

    curves([r for r in monitor if r['estimator'] == 'knn_static'], 'strict_accuracy',
           'GSM8K accuracy during PPO', 'fig_ppo_accuracy_trajectory', 'Strict accuracy')
    curves(owned, 'optimism_mean', 'Mean gap optimism under the trained policies', 'fig_optimism_trajectory', 'Mean actual gap - predicted gap')
    curves(owned, 'optimism_p95', 'Upper tail of gap optimism', 'fig_optimism_p95_trajectory', '95th percentile optimism')
    curves([r for r in monitor if r['estimator'] == 'knn_static'], 'mean_nn_similarity',
           'Similarity to the frozen fitting memory', 'fig_memory_similarity_trajectory', 'Mean nearest cosine similarity')
    static = [r for r in tables['gap_prediction'] if r['cohort'] == 'final' and r['arm'] == 'base']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(axes, ('gap_mse', 'gap_r2', 'high_gap_auroc'), ('Gap MSE', 'Gap R2', 'High-gap AUROC')):
        selected = [r for r in static if r.get(metric) is not None]
        ax.bar(range(len(selected)), [r[metric] for r in selected])
        ax.set_xticks(range(len(selected)), [f"{r['estimator']}\ns{r['seed']}" for r in selected], rotation=25, fontsize=7)
        ax.set_title(title+' on shared base answers')
    finish(fig, 'fig_gap_prediction')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, estimator in zip(axes, ('knn_static', 'ridge')):
        grouped = defaultdict(list)
        for r in owned:
            if r['estimator'] == estimator:
                grouped[r['seed']].append(r)
        for seed, values in grouped.items():
            initial = base.get((seed, estimator))
            values = ([initial] if initial else [])+sorted(values, key=lambda r: r['step'])
            for key, style in (('mean_true_gap', '-'), ('mean_pred_gap', '--')):
                selected = [r for r in values if r.get(key) is not None]
                ax.plot([r['step'] for r in selected], [r[key] for r in selected], style, label=f'{key}, seed {seed}')
        ax.set(title=estimator, xlabel='PPO attempt (monitor)', ylabel='Normalized gap')
    finish(fig, 'fig_gap_true_vs_pred')
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped = defaultdict(list)
    for r in owned:
        if r['optimism_mean'] is not None and r['strict_accuracy'] is not None:
            grouped[r['seed'], r['arm']].append(r)
    for (seed, arm), rows in grouped.items():
        ax.scatter([r['optimism_mean'] for r in rows], [r['strict_accuracy'] for r in rows], label=f'{arm}, seed {seed}')
    ax.set(xlabel='Mean optimism', ylabel='Strict accuracy', title='Each point is a monitoring checkpoint')
    finish(fig, 'fig_accuracy_vs_optimism')
    for metric, title, filename in (('gap_mse', 'Gap MSE', 'fig_error_vs_memory_distance'),
                                    ('optimism_mean', 'Mean optimism', 'fig_optimism_vs_memory_distance')):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, arm in zip(axes, ('base', 'knn_static', 'ridge')):
            grouped = defaultdict(list)
            for r in tables['distribution_shift']:
                if r['cohort'] == 'final' and r['arm'] == arm and r['estimator'] != 'mean_gap' and r[metric] is not None:
                    grouped[r['seed'], r['estimator']].append(r)
            for (seed, estimator), rows in grouped.items():
                rows = sorted(rows, key=lambda r: r['distance_quintile'])
                ax.plot([r['mean_nn_distance'] for r in rows], [r[metric] for r in rows], '.-', label=f'{estimator}, s{seed}')
            ax.set(title=f'{arm} final answers', xlabel='Mean nearest-memory distance', ylabel=title)
        finish(fig, filename)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, arm in zip(axes, ('base', 'knn_static', 'ridge')):
        grouped = defaultdict(list)
        for r in tables['top_reward_accuracy']:
            if r['cohort'] == 'final' and r['arm'] == arm and r['correctness'] == 'correct' and r['accuracy'] is not None and r['reward_signal'] != 'mean_gap':
                grouped[r['seed'], r['reward_signal']].append(r)
        for (seed, signal), rows in grouped.items():
            rows = sorted(rows, key=lambda r: r['top_fraction'])
            ax.plot([100*r['top_fraction'] for r in rows], [r['accuracy'] for r in rows], '.-', label=f'{signal}, s{seed}')
        ax.set(xscale='log', title=f'{arm} final answers', xlabel='Nominal top reward % (ties retained)', ylabel='Strict correctness', ylim=(0, 1))
        ax.set_xticks([1, 5, 10, 20, 50, 100], ['1', '5', '10', '20', '50', '100'])
    finish(fig, 'fig_top_reward_accuracy')
    rows = [r for r in tables['length_shortcuts'] if r['cohort'] == 'final' and r['target'] == 'predicted_gap' and r['estimator'] != 'mean_gap' and r['spearman'] is not None]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(rows)), [r['spearman'] for r in rows])
    ax.set_xticks(range(len(rows)), [f"{r['arm']} / {r['estimator']}\ns{r['seed']}" for r in rows], rotation=45, fontsize=6)
    ax.set(title='Length versus predicted gap on final answers', ylabel='Spearman correlation')
    finish(fig, 'fig_length_gap_correlation')
