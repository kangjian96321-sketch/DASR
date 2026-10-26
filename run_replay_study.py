"""Run each seed's DTR anchor, then three controls with its replay activation mask."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
from ablation_config import ABLATIONS, method_name
from far_replay import SOURCE_MODES
from my_functions.backbone_options import add_backbone_argument, backbone_suffix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_backbone_argument(p)
    p.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100', 'tiny_imagenet', 'imagenette'])
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--data-root', default='./data')
    p.add_argument('--output-root', default='./runs_replay')
    p.add_argument('--run-name', default='pace_v3')
    p.add_argument('--ablation', choices=ABLATIONS, default='full')
    p.add_argument('--replay-source-mode', choices=SOURCE_MODES, default='fg_adaptive_lazy')
    p.add_argument('--resume-mode', choices=['task','epoch'], default='task')
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--skip-autoattack', action='store_true')
    p.add_argument('--exclude-clean', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.ablation in ('only_cls','no_dtr'):
        p.error('This preset has no replay. Use run_ablations.py for ablation studies.')
    if len(set(args.seeds)) != len(args.seeds):
        p.error('Each training seed must appear exactly once')
    entry = Path(__file__).with_name('train_replay_comparison.py')
    output = Path(args.output_root).resolve()
    for seed in args.seeds:
        anchor_method = f'{method_name("dtr", args.ablation)}_{args.replay_source_mode}'
        suffix = '_' + args.run_name if args.run_name else ''
        architecture = backbone_suffix(args.backbone)
        anchor = output / f'{anchor_method}{architecture}_{args.dataset}_seed{seed}{suffix}' / 'replay_schedule.json'
        for policy in ['dtr', 'random', 'uniform', 'lowest_accuracy']:
            cmd = [sys.executable, str(entry), '--policy', policy, '--dataset', args.dataset,
                   '--backbone', args.backbone,
                   '--seed', str(seed), '--data-root', str(Path(args.data_root).resolve()),
                   '--output-root', str(output), '--run-name', args.run_name,
                   '--device', args.device, '--workers', str(args.workers),
                   '--ablation', args.ablation, '--resume-mode', args.resume_mode]
            cmd += ['--replay-source-mode', args.replay_source_mode]
            if policy != 'dtr':
                cmd += ['--match-schedule', str(anchor)]
            if args.batch_size:
                cmd += ['--batch-size', str(args.batch_size)]
            method = f'{method_name(policy,args.ablation)}_{args.replay_source_mode}'
            run = output/f'{method}{architecture}_{args.dataset}_seed{seed}{suffix}'
            if args.resume and run.exists() and any(run.glob('*.pt')):
                cmd += ['--resume']
            for flag in ['smoke', 'skip_autoattack', 'exclude_clean']:
                if getattr(args, flag):
                    cmd += ['--' + flag.replace('_', '-')]
            print(shlex.join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
