"""Validate matched budgets and export measured mean/sample SD as CSV and LaTeX.

Refuses incomplete runs, synthetic smoke data, mismatched protocols, and missing
AutoAttack scores. A missing result is never silently turned into zero.
"""
import argparse
import csv
import copy
import json
import math
from pathlib import Path
from statistics import mean, stdev
from ablation_config import ABLATIONS, method_name

POLICIES = ['dtr', 'random', 'uniform', 'lowest_accuracy']
METRICS = ['clean', 'patch', 'pgd_l2', 'pgd', 'autoattack_l2', 'autoattack']
LABELS = {'dtr': 'DTR', 'random': 'Random replay', 'uniform': 'Uniform replay',
          'lowest_accuracy': 'Lowest-accuracy replay'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def collect(root, dataset, run_name, seeds, ablation='full'):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('Use at least two distinct seeds for a sample standard deviation')
    data = {policy: [] for policy in POLICIES}
    resources = []
    verified_settings = None
    for seed in seeds:
        anchor_dir = root / f'{method_name("dtr", ablation)}_{dataset}_seed{seed}_{run_name}'
        anchor_schedule = read(anchor_dir / 'replay_schedule.json')
        anchor_mask = [(r['stage'], r['epoch'], r['active']) for r in anchor_schedule['epochs']]
        anchor_counts = [(r['stage'], r['replay_examples'], r['replay_batches'])
                         for r in read(anchor_dir / 'resource_metrics.json')]
        anchor_split = read(anchor_dir / 'data_split.json')['signature']
        for policy in POLICIES:
            run = root / f'{method_name(policy, ablation)}_{dataset}_seed{seed}_{run_name}'
            completion = read(run / 'run_complete.json')
            if completion.get('status') != 'experiment':
                raise ValueError(f'Not a completed real experiment: {run}')
            config = read(run / 'comparison_config.json')
            if completion.get('imported_history') or not completion.get('history_complete', True):
                raise ValueError('Matched studies require complete measured histories; export individual runs instead')
            comparable = copy.deepcopy(config['settings'])
            comparable['config'].pop('seed')
            comparable['evaluation'] = read(run/'final_test_meta.json').get('evaluation_settings')
            if verified_settings is None:
                verified_settings = comparable
            elif comparable != verified_settings:
                raise ValueError('Different configurations across seeds; use individual results')
            schedule = read(run / 'replay_schedule.json')
            if config['identity']['protocol_signature'] != anchor_schedule['protocol_signature']:
                raise ValueError('Protocol mismatch across replay arms')
            if read(run / 'data_split.json')['signature'] != anchor_split:
                raise ValueError('Different data splits across replay arms')
            mask = [(r['stage'], r['epoch'], r['active']) for r in schedule['epochs']]
            if not schedule['complete'] or not schedule.get('history_complete', True) or mask != anchor_mask:
                raise ValueError('Replay activation masks are not matched')
            if policy != 'dtr' and schedule['gate'] != 'matched':
                raise ValueError('Alternative is not a matched-budget run')
            rows = read(run / 'resource_metrics.json')
            counts = [(r['stage'], r['replay_examples'], r['replay_batches']) for r in rows]
            if counts != anchor_counts:
                raise ValueError('Actual replay example/batch budgets differ')
            result = read(run / 'final_test.json')
            for metric in METRICS:
                if metric not in result or not math.isfinite(result[metric]) or not 0 <= result[metric] <= 100:
                    raise ValueError(f'Missing/non-finite measured {metric} in {run}')
            values = {key: float(result[key]) for key in METRICS}
            values['mean'] = mean(values.values())
            values['minimum'] = min(values[key] for key in METRICS)
            # Validation forgetting is explicitly distinguished from test forgetting.
            matrix = read(run / 'accuracy_matrix.json')
            final = matrix[-1]['accuracies']
            old = list(config['settings']['config']['task_sequence'])[:-1]
            forgetting = []
            for task in old:
                best_past = max(row['accuracies'][task] for row in matrix[:-1] if task in row['accuracies'])
                forgetting.append(best_past - final[task])
            values['validation_forgetting_pp'] = mean(forgetting) if forgetting else 0.0
            data[policy].append(values)
            resources.extend({'policy': policy, 'seed': seed, **row} for row in rows)
    return data, resources


def aggregate_resources(resources):
    """Mean/sample SD by policy and stage. Never sum peak memory across stages."""
    if not resources:
        return []
    keys = sorted({key for row in resources for key in row if key.endswith(('_seconds', '_bytes', '_samples'))})
    keys += ['replay_examples', 'clean_replay_examples', 'replay_batches', 'active_replay_epochs']
    groups = {}
    for row in resources:
        groups.setdefault((row['policy'], row['stage'], row['task']), []).append(row)
    summaries = []
    for (policy, stage, task), rows in groups.items():
        summary = {'policy': policy, 'stage': stage, 'task': task, 'n': len(rows)}
        for key in keys:
            numbers = [row.get(key) for row in rows]
            # CPU runs have no GPU elapsed time; preserve unavailable as blank.
            available = all(isinstance(v, (int, float)) and math.isfinite(v) for v in numbers)
            summary[key + '_mean'] = mean(numbers) if available else None
            summary[key + '_std'] = stdev(numbers) if available and len(numbers) > 1 else None
        summaries.append(summary)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, default=Path('./runs_replay'))
    p.add_argument('--dataset', default='cifar100')
    p.add_argument('--run-name', default='pace_v3')
    p.add_argument('--ablation', choices=ABLATIONS, default='full')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = p.parse_args()
    data, resources = collect(args.output_root, args.dataset, args.run_name, args.seeds, args.ablation)
    out = args.output_root / 'summaries'
    out.mkdir(exist_ok=True)
    all_metrics = METRICS + ['mean', 'minimum', 'validation_forgetting_pp']
    summaries = []
    latex_rows = []
    for policy, values in data.items():
        row = {'policy': policy, 'n': len(values)}
        cells = []
        for metric in all_metrics:
            numbers = [v[metric] for v in values]
            row[metric + '_mean'], row[metric + '_std'] = mean(numbers), stdev(numbers)
            if metric != 'validation_forgetting_pp':
                cells.append(f'${mean(numbers):.2f} \\pm {stdev(numbers):.2f}$')
        summaries.append(row)
        latex_rows.append(LABELS[policy] + ' & ' + ' & '.join(cells) + r' \\')
    stem = args.dataset + '_' + args.run_name
    for suffix, rows in [('summary', summaries), ('resources_by_seed_stage', resources),
                         ('resources_mean_std_by_stage', aggregate_resources(resources))]:
        with (out / f'{stem}_{suffix}.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
    (out / f'{stem}_table_rows.tex').write_text('\n'.join(latex_rows) + '\n', encoding='utf-8')
    print('Verified all matched budgets; exported measured results to', out)


if __name__ == '__main__':
    main()
