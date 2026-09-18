import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from workshop.common import DEFAULT_CONFIG, ROOT, atomic_json, load_config, read_json
from workshop import suite


def test_copied_project_runs_without_original_repository(tmp_path):
    isolated = tmp_path/'standalone'
    shutil.copytree(ROOT/'workshop', isolated/'workshop', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'configs', isolated/'configs')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    command = [sys.executable, '-m', 'workshop', 'run', '--seeds', '42', '43', '44', '--dry-run']
    result = subprocess.run(command, cwd=isolated, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['arms'] == ['proxy', 'judge', 'knn_static', 'knn_static_30b', 'ridge']
    assert plan['seeds'] == [42, 43, 44]
    assert plan['attempts_per_arm'] == 400 and plan['memory_answers_before_exclusions'] == 1024
    assert not (isolated/'outputs').exists()
    for bad in (['--seeds', '42', '42'], ['--updates', '0']):
        result = subprocess.run([*command[:5], *bad, '--dry-run'], cwd=isolated, env=env, capture_output=True, text=True)
        assert result.returncode != 0


@pytest.mark.parametrize('profile,batches', [('b200', (256, 128, 32)), ('b300', (512, 192, 64))])
def test_runtime_profiles_share_experiment_definition(tmp_path, profile, batches):
    # An isolated copy also proves inheritance does not depend on the cwd.
    shutil.copytree(ROOT/'configs', tmp_path/'configs')
    base = load_config(tmp_path/'configs/default.yaml')
    config = load_config(tmp_path/'configs'/f'{profile}.yaml')
    assert 'extends' not in config
    for section, expected in zip(('generation', 'scoring', 'teacher30b'), batches):
        assert config[section]['batch_size'] == expected
        config[section]['batch_size'] = base[section]['batch_size']
    assert config == base
    assert load_config(tmp_path/'configs/default.yaml') == base


def test_configuration_inheritance_rejects_cycles_and_bad_parents(tmp_path):
    left, right = tmp_path/'left.yaml', tmp_path/'right.yaml'
    left.write_text('extends: right.yaml\n', encoding='utf-8')
    right.write_text('extends: left.yaml\n', encoding='utf-8')
    with pytest.raises(ValueError, match='inheritance cycle'):
        load_config(left)
    left.write_text('extends: 42\n', encoding='utf-8')
    with pytest.raises(ValueError, match='parent file'):
        load_config(left)
    left.write_text('extends: right.yaml\n', encoding='utf-8')
    right.write_text('- not a mapping\n', encoding='utf-8')
    with pytest.raises(ValueError, match='must be a mapping'):
        load_config(left)


def test_seed_suite_pins_assets_and_can_append_seeds(tmp_path, monkeypatch):
    config = load_config(DEFAULT_CONFIG)
    original = copy.deepcopy(config)
    calls = []
    def worker(command, **kwargs):
        current = read_json(Path(command[command.index('--config')+1]))
        output = Path(command[command.index('--output')+1])
        calls.append((current['seed'], current['data_seed'], current['arms']))
        pinned = output/'resolved_assets.json'
        if current['seed'] == 42:
            atomic_json(pinned, {'dataset': 'frozen', 'policy': 'frozen'})
        else:
            assert read_json(pinned) == {'dataset': 'frozen', 'policy': 'frozen'}
        assert kwargs['cwd'] == ROOT
        return 0
    monkeypatch.setattr(suite.subprocess, 'call', worker)
    suite.run_suite(config, tmp_path, [42], 'prepare')
    suite.run_suite(config, tmp_path, [42, 43, 44], 'prepare')
    assert calls == [(42, 42, config['arms']), (42, 42, config['arms']), (43, 42, config['arms']), (44, 42, config['arms'])]
    assert config == original
    assert (tmp_path/'seed_additions.jsonl').exists()
    changed = copy.deepcopy(config)
    changed['ridge']['alphas'] = [1.]
    with pytest.raises(ValueError, match='configuration'):
        suite.run_suite(changed, tmp_path, [42, 43, 44], 'prepare')


def test_suite_reports_ridge_and_individual_seed_differences(tmp_path):
    arms = ['proxy', 'knn_static', 'ridge']
    for seed, delta in ((42, .1), (43, .2)):
        folder = tmp_path/f'seed_{seed}'
        atomic_json(folder/'final_protocol.json', {'arms': arms})
        metrics = [{'arm': arm, 'cohort': 'final', 'accuracy': .4+(delta if arm == 'ridge' else 0.),
                    'numeric_accuracy': .5+(delta if arm == 'ridge' else 0.),
                    'numeric_unresolved_rate': .1, 'format_valid_rate': .8, 'length_cap_rate': .05}
                   for arm in ['base', *arms]]
        atomic_json(folder/'summary.json', {'metrics': metrics})
        atomic_json(folder/'predictors/summary.json', {'metrics': [
            {'seed': seed, 'cohort': 'final/base/0', 'predictor': 'ridge', 'gap_mse': 1.-delta,
             'optimistic_tail_bias_01': delta, 'optimistic_tail_bias_01_n': 2 if seed == 42 else 4}]})
    suite.aggregate(tmp_path, [42, 43], arms)
    result = read_json(tmp_path/'suite_summary.json')
    assert result['arms']['ridge']['accuracy']['mean'] == pytest.approx(.55)
    assert result['paired_differences']['ridge_minus_knn_static']['strict_difference']['mean'] == pytest.approx(.15)
    assert [r['seed'] for r in result['per_seed_differences']['ridge_minus_proxy']] == [42, 43]
    row = next(r for r in result['predictors'] if r['metric'] == 'gap_mse')
    assert row['mean'] == pytest.approx(.85)
    assert (tmp_path/'suite_predictors.csv').exists()
    tail = next(r for r in result['predictors'] if r['metric'] == 'optimistic_tail_bias_01')
    assert tail['mean'] == pytest.approx(.15) and tail['n_seeds'] == 2
    count = next(r for r in result['predictors'] if r['metric'] == 'optimistic_tail_bias_01_n')
    assert count['mean'] == 3.
    assert 'OTB 1% (n)' in (tmp_path/'suite_report.md').read_text(encoding='utf-8')
