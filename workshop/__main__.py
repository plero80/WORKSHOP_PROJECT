"""The single user entry point. Dry runs never import model or PPO code."""
import argparse
import json
from pathlib import Path

from .common import ARMS, DEFAULT_CONFIG, OUTPUT_ROOT, ROOT, load_config, read_json, validate_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m workshop',
        description='Workshop kNN: matched proxy, judge, kNN and ridge PPO on GSM8K.')
    actions = parser.add_subparsers(dest='action', required=True)
    run = actions.add_parser('run', help='Start or resume all declared reward arms.')
    run.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    run.add_argument('--output', type=Path, default=OUTPUT_ROOT/'main')
    run.add_argument('--seeds', type=int, nargs='+', help='Default: the seed in the YAML. Example: 42 43 44')
    run.add_argument('--stage', choices=['prepare', 'pilot', 'full'], default='full')
    run.add_argument('--arms', nargs='+', choices=ARMS, help='Override the YAML arm list before starting a new run.')
    run.add_argument('--updates', type=int, help='Override the total target, not the number of additional attempts.')
    run.add_argument('--dry-run', action='store_true', help='Print the plan; no files, downloads or training.')
    for name in ('report', 'status'):
        sub = actions.add_parser(name)
        sub.add_argument('--output', type=Path, default=OUTPUT_ROOT/'main')
    analysis = actions.add_parser('analyze', help='Generate paper tables, figures and statistics from saved answers; CPU only.')
    analysis.add_argument('--output', type=Path, default=OUTPUT_ROOT/'main', help='A seed-suite or single-seed output directory.')
    analysis.add_argument('--destination', type=Path, help='Default: OUTPUT/paper_results')
    analysis.add_argument('--bootstrap-samples', type=int, help='Override the saved evaluation bootstrap count (0 disables intervals).')
    analysis.add_argument('--no-plots', action='store_true', help='Generate tables/statistics without PDF/PNG figures.')
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        if args.action == 'run':
            config = load_config(args.config)
            if args.arms is not None:
                config['arms'] = args.arms
            validate_config(config)
            seeds = args.seeds or [config['seed']]
            if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
                raise ValueError('Seeds must be distinct integers in [0, 2**32).')
            if args.updates is not None and args.updates < 1:
                raise ValueError('--updates must be positive.')
            for name in ('workshop', 'configs', 'tests', 'scripts'):
                protected = ROOT/name
                if output.is_relative_to(protected) or protected.is_relative_to(output):
                    raise ValueError('Choose a separate output directory, outside code/configs/tests.')
            target = args.updates or config['ppo']['pilot_updates' if args.stage == 'pilot' else 'full_updates']
            p, d = config['ppo'], config['dataset']
            print(json.dumps({'output': str(output), 'seeds': seeds, 'stage': args.stage,
                'arms': config['arms'], 'attempts_per_arm': 0 if args.stage == 'prepare' else target,
                'rollout_answers_per_arm': 0 if args.stage == 'prepare' else target*p['prompts_per_update']*p['responses_per_prompt'],
                'memory_questions': d['memory'], 'memory_answers_before_exclusions': d['memory']*d['responses_per_prompt'],
                'ridge_target': 'actual normalized 4B gap, same memory as static kNN',
                'ridge_selection': 'separate validation MSE, frozen before PPO',
                'final_questions_per_policy': d['final'] if args.stage == 'full' else 0,
                'runtime_batches': {'generation': config['generation']['batch_size'],
                                    'grading': config['scoring']['batch_size'],
                                    'teacher30b': config.get('teacher30b', {}).get('batch_size'),
                                    'ppo_microbatch': config['runtime']['ppo_microbatch_size']},
                'engine': 'one workshop/ppo.py for every arm'}, indent=2), flush=True)
            if args.dry_run:
                return 0
            from .suite import run_suite
            run_suite(config, output, seeds, args.stage, args.updates)
        elif args.action == 'analyze':
            from .analysis import analyze
            analyze(output, args.destination, bootstrap_samples=args.bootstrap_samples, figures=not args.no_plots)
        elif args.action == 'status':
            path = output/'suite_protocol.json'
            seeds = read_json(path)['seeds'] if path.exists() else []
            paths = [output/'suite_status.json', *[output/f'seed_{s}/status.json' for s in seeds]]
            found = {str(p.relative_to(output)): read_json(p) for p in paths if p.exists()}
            if not found:
                raise ValueError(f'No saved status under {output}')
            print(json.dumps(found, indent=2))
        else:
            from .report import make_report
            from .suite import aggregate
            path = output/'suite_protocol.json'
            if path.exists():
                plan = read_json(path)
                for seed in plan['seeds']:
                    make_report(output/f'seed_{seed}')
                if all((output/f'seed_{s}/final_protocol.json').exists() for s in plan['seeds']):
                    aggregate(output, plan['seeds'], plan['config']['arms'])
            else:
                make_report(output)
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
