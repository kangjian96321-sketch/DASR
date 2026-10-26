"""Export one row per run; optionally aggregate verified identical configurations."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

METRICS = ('clean','patch','pgd_l2','pgd','autoattack_l2','autoattack')


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def collect(root, include_smoke=False):
    rows = []
    for marker in sorted(root.rglob('run_complete.json')):
        if any(x in ('backups','imported_originals') for x in marker.relative_to(root).parts):
            continue
        run = marker.parent
        done = read(marker)
        if done.get('status') != 'experiment' and not include_smoke:
            continue
        config = read(run/'comparison_config.json')
        if not config:
            continue
        settings = copy.deepcopy(config['settings'])
        seed = settings['config'].pop('seed')
        evaluation = read(run/'final_test_meta.json', {})
        # The fingerprint contains weights and seed, so compare only its explicit
        # evaluation protocol; legacy results without one stay individual.
        evaluation_settings = evaluation.get('evaluation_settings')
        imported = bool(done.get('imported_history') or config.get('imported_history'))
        verified = not imported and evaluation_settings is not None
        grouping = {'settings': settings, 'policy': config['identity']['policy'],
                    'gate': config['identity']['gate'], 'evaluation': evaluation_settings,
                    'split': read(run/'data_split.json', {}).get('signature')}
        if not verified:
            grouping['unverified_run'] = str(run.resolve())
        group_id = hashlib.sha256(json.dumps(grouping, sort_keys=True).encode()).hexdigest()[:16]
        result = read(run/'final_test.json', {})
        row = {'run': str(run.resolve()), 'ablation': settings.get('ablation', {}).get('name','full'),
               'dataset': settings['config']['dataset'], 'policy': config['identity']['policy'],
               'seed': seed, 'configuration_group': group_id, 'config_verified': verified,
               'imported_history': imported, 'history_complete': done.get('history_complete', False),
               'status': done['status']}
        for metric in (*METRICS, 'cw_l2'):
            value = result.get(metric)
            row[metric] = value if number(value) and 0 <= value <= 100 else None
        all_values = [row[key] for key in METRICS]
        row['mean_six'] = mean(all_values) if all(number(x) for x in all_values) else None
        row['minimum_six'] = min(all_values) if row['mean_six'] is not None else None
        matrix = read(run/'accuracy_matrix.json', [])
        old_tasks = settings['config']['task_sequence'][:-1]
        drops = []
        if matrix and len(matrix) == len(settings['config']['task_sequence']):
            for task in old_tasks:
                prior = [r['accuracies'][task] for r in matrix[:-1]
                         if r.get('accuracies') is not None and number(r['accuracies'].get(task))]
                final = (matrix[-1].get('accuracies') or {}).get(task)
                expected_count = len(matrix)-1-settings['config']['task_sequence'].index(task)
                if len(prior) == expected_count and number(final):
                    drops.append(max(prior)-final)
        row['validation_forgetting_pp'] = mean(drops) if old_tasks and len(drops) == len(old_tasks) else None
        rows.append(row)
    return rows


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['configuration_group'], []).append(row)
    output = []
    for key, values in groups.items():
        if not all(r['config_verified'] for r in values) or len(values) < 2:
            continue
        if len({r['seed'] for r in values}) != len(values):
            raise ValueError(f'Duplicate training seed in configuration group {key}; choose one run per seed')
        result = {k: values[0][k] for k in ('ablation','dataset','policy','configuration_group')}
        result['n'] = len(values)
        for metric in (*METRICS,'cw_l2','mean_six','minimum_six','validation_forgetting_pp'):
            numbers = [r[metric] for r in values]
            available = all(number(x) for x in numbers)
            result[metric+'_mean'] = mean(numbers) if available else None
            result[metric+'_std'] = stdev(numbers) if available else None
        output.append(result)
    return output


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, default=Path('./runs_replay'))
    p.add_argument('--save-dir', type=Path)
    p.add_argument('--aggregate', action='store_true')
    p.add_argument('--include-smoke', action='store_true')
    args = p.parse_args(argv)
    rows = collect(args.output_root, args.include_smoke)
    if not rows:
        raise ValueError('No completed eligible runs found')
    out = args.save_dir or args.output_root/'ablation_summary'; out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'individual_runs.csv', rows)
    lines = []
    for row in rows:
        cells = ['--' if row[k] is None else f'{row[k]:.2f}' for k in METRICS]
        lines.append(row['ablation'].replace('_',r'\_')+f" (seed {row['seed']}) & "+' & '.join(cells)+r' \\')
    (out/'individual_table_rows.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if args.aggregate:
        grouped = aggregate(rows)
        write_csv(out/'verified_seed_statistics.csv', grouped)
        (out/'aggregation_report.json').write_text(json.dumps({'individual_runs': len(rows),
            'verified_groups_with_multiple_seeds': len(grouped),
            'rule': 'Same training/evaluation protocol; only training seed differs; imported history excluded'},
            indent=2), encoding='utf-8')
    print('Saved:', out)


if __name__ == '__main__':
    main()
