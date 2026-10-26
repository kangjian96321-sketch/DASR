"""Run the four FAR replay-source experiments sequentially for each seed."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from ablation_config import ABLATIONS, method_name
from far_replay import SOURCE_MODES
from my_functions.backbone_options import add_backbone_argument, backbone_suffix


def build_commands(args, extra):
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be unique")
    if len(args.modes) != len(set(args.modes)):
        raise ValueError("Replay source modes must be unique")
    forbidden = {
        "--run-dir", "--seed", "--policy", "--ablation",
        "--replay-source-mode", "--resume", "--backbone",
    }
    if any(value.split("=")[0] in forbidden for value in extra):
        raise ValueError("Use this launcher's identity options")
    entry = Path(__file__).with_name("train_DASR.py")
    backbone = getattr(args, "backbone", "wrn")
    commands = []
    for seed in args.seeds:
        for mode in args.modes:
            command = [
                sys.executable, str(entry),
                "--policy", "dtr",
                "--ablation", args.ablation,
                "--replay-source-mode", mode,
                "--dataset", args.dataset,
                "--backbone", backbone,
                "--data-root", args.data_root,
                "--output-root", args.output_root,
                "--run-name", args.run_name,
                "--seed", str(seed),
                *extra,
            ]
            method = f"{method_name('dtr', args.ablation)}_{mode}"
            run_dir = (
                Path(args.output_root)
                / f"{method}{backbone_suffix(backbone)}_{args.dataset}_seed{seed}"
                  f"{'_' + args.run_name if args.run_name else ''}"
            )
            if args.resume and run_dir.exists() and any(run_dir.glob("*.pt")):
                command += ["--resume", "--resume-mode", args.resume_mode]
            commands.append(command)
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_backbone_argument(parser)
    parser.add_argument("--modes", choices=SOURCE_MODES, nargs="+",
                        default=list(SOURCE_MODES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--dataset", choices=[
        "cifar10", "cifar100", "tiny_imagenet", "imagenette"
    ], default="cifar100")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-root", default="./runs_replay")
    parser.add_argument("--run-name", default="far_four_sources")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-mode", choices=["task", "epoch"], default="epoch")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    for command in build_commands(args, extra):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
