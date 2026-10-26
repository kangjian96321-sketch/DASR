"""Run ablations sequentially. Extra arguments are forwarded to the trainer."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from ablation_config import ABLATIONS, method_name
from far_replay import SOURCE_MODES
from my_functions.backbone_options import add_backbone_argument, backbone_suffix


def build_commands(args, extra):
    if len(args.seeds) != len(set(args.seeds)) or len(args.ablations) != len(set(args.ablations)):
        raise ValueError('Seeds and ablations must be unique')
    forbidden = {'--run-dir', '--seed', '--ablation', '--policy', '--resume',
                 '--replay-source-mode', '--backbone'}
    if any(x.split('=')[0] in forbidden for x in extra):
        raise ValueError('Use this launcher options for run identity and checkpoint sources')
    mapping = json.loads(args.checkpoint_map.read_text(encoding='utf-8')) if args.checkpoint_map else {}
    entry = Path(__file__).with_name('train_replay_comparison.py')
    # Keep programmatic callers of the pre-FAR launcher working.
    replay_source_mode = getattr(args, 'replay_source_mode', 'fg_adaptive_lazy')
    backbone = getattr(args, 'backbone', 'wrn')
    commands = []
    for seed in args.seeds:
        for ablation in args.ablations:
            method = f'{method_name(args.policy, ablation)}_{replay_source_mode}'
            run = Path(args.output_root)/(f'{method}{backbone_suffix(backbone)}_{args.dataset}_seed{seed}'
                                          f'{"_" + args.run_name if args.run_name else ""}')
            cmd = [sys.executable, str(entry), '--ablation', ablation, '--policy', args.policy,
                   '--replay-source-mode', replay_source_mode,
                   '--backbone', backbone,
                   '--seed', str(seed), '--dataset', args.dataset, '--data-root', args.data_root,
                   '--output-root', args.output_root, '--run-name', args.run_name, *extra]
            key = f'{ablation}:{seed}'
            source = mapping.get(key, mapping.get(ablation))
            if source and key not in mapping and len(args.seeds) > 1:
                raise ValueError('For multiple seeds use checkpoint-map keys such as no_queue:42')
            if source:
                if isinstance(source, str):
                    source = {'path': source}
                cmd += ['--resume', source['path']]
                for name in ('completed_through', 'start_task', 'checkpoint_task', 'resume_mode'):
                    if name in source:
                        flag = '--'+name.replace('_', '-')
                        if any(v.split('=')[0] == flag for v in extra):
                            raise ValueError(f'Duplicate setting {flag} in map and command line')
                        cmd += [flag, str(source[name])]
            elif args.resume and run.exists() and any(run.glob('*.pt')):
                cmd += ['--resume']
            elif args.resume and run.exists() and any(run.glob('*.pth')):
                cmd += ['--resume']
            commands.append(cmd)
    return commands


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_backbone_argument(p)
    p.add_argument('--ablations', choices=ABLATIONS, nargs='+', default=list(ABLATIONS))
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--policy', choices=['dtr','random','uniform','lowest_accuracy','none'], default='dtr')
    p.add_argument('--replay-source-mode', choices=SOURCE_MODES, default='fg_adaptive_lazy')
    p.add_argument('--dataset', default='cifar100')
    p.add_argument('--data-root', default='./data')
    p.add_argument('--output-root', default='./runs_replay')
    p.add_argument('--run-name', default='pace_v3')
    p.add_argument('--checkpoint-map', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args, extra = p.parse_known_args(argv)
    for command in build_commands(args, extra):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
