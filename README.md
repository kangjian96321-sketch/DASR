# DASR / PACE: continual adversarial training

[中文说明](README_zh.md) · [Protocol and parameters](docs/protocol.md) · [Changes](docs/changes.md) · [GitHub upload guide](docs/github_upload_zh.md)

Research implementation of continual adversarial training with task selection,
historical-versus-regenerated replay, feature distillation, and comparison baselines.
The legacy `PACE` module and checkpoint names are retained. This release contains
PACE/DASR, SSEAT-aligned, CRT-aligned, TRADES, MART, DER++, GSA, and Joint-Mean.
It contains neither CAD nor the standalone Patch-stage restart utility.

## Release definition

This release extends the 20261022 four-task project. It changes **all trainers** to
select the best checkpoint only after their scheduler's warmup, includes zero
accuracies in selection means, sets FAR monitor/history defaults to **500/500**,
and integrates the later official-aligned L2/Linf CRT ALR module. It is a new
training protocol: **start new experiment directories; do not reuse old results
as results of this release**.

Historical FAR examples still come from each attack's original selected model
and persist across stages. Updating the preceding-stage accuracy reference does
not regenerate those examples. The stage-refreshed FAR branch is not included.

## 1. Environment

Run commands from this repository root. Python 3.12 and the torch 2.4.1 /
torchvision 0.19.1 pair are the regression target. Choose one setup:

```bash
# NVIDIA / CUDA 12.1 recipe (Linux or Windows)
conda env create -f environment.yml
conda activate dasr-pace

# Alternative CPU-only regression environment
conda env create -f environment-cpu.yml
conda activate dasr-pace-cpu
```

Alternatively, inside an existing isolated Python environment:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-dev.txt
```

Replace `cu121` with `cpu` for CPU tests. These are reproduction pins, not a claim
to use the newest PyTorch. For other compatible CUDA builds, consult the
[official version-specific installation instructions](https://pytorch.org/get-started/previous-versions/).
The YAML files are environment recipes, not fully resolved lockfiles. The
actually tested runtime is recorded in `docs/test_environment.json`; local CPU
verification is not a CUDA/Conda installation verification.

The Conda YAML recipes already include the
[official AutoAttack package](https://github.com/fra31/auto-attack) as a Git dependency.
For the manual pip setup, install it separately:

```bash
python -m pip install -r requirements-autoattack.txt
python -m pip freeze
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Keep the resolved AutoAttack revision and package list with each experiment;
the install command follows upstream and is not a pinned commit. The source
package does not bundle datasets, weights, AutoAttack, or a CRT checkout.

## 2. Data

- `--dataset cifar10` / `cifar100`: torchvision downloads to `--data-root`.
- `--dataset imagenette`: prepare an Imagenette-160 ImageFolder tree below.
- `--dataset tiny_imagenet`: use `tiny-imagenet-200/train/` and the official
  validation annotations, or class-organized validation folders.

```text
data/imagenette2-160/
  train/<class_id>/*.JPEG
  val/<class_id>/*.JPEG
```

For Imagenette, a seeded subset of `train/` is held out for model selection;
the provided `val/` is the final test set. The default held-out size is 1,000
(5,000 for the other datasets). No test labels are used for training or best
selection. The split is seeded, not globally class-stratified; the monitor is
a class-balanced subset of that split. Asking for more examples of a class than
exist raises an error instead of silently duplicating examples.

## 3. Both task orders

`pgd` means PGD-Linf, and `pgd_l2` means PGD-L2.

| Method | Clean → Patch → L2 → Linf | Clean → Linf → L2 → Patch |
|---|---|---|
| PACE/DASR | `train_replay_comparison.py` | `train_pace_clean_linf_l2_patch.py` |
| SSEAT | `train_sseat_pace_aligned.py` | `train_sseat_clean_linf_l2_patch.py` |
| CRT | `train_crt_pace_aligned.py` | `train_crt_clean_linf_l2_patch.py` |
| TRADES | `train_trades_clean_patch_l2_linf.py` | `train_trades_clean_linf_l2_patch.py` |
| MART | `train_mart_clean_patch_l2_linf.py` | `train_mart_clean_linf_l2_patch.py` |
| DER++ | `train_derpp_clean_patch_l2_linf.py` | `train_derpp_clean_linf_l2_patch.py` |
| GSA | `train_gsa_clean_patch_l2_linf.py` | `train_gsa_clean_linf_l2_patch.py` |
| Joint-Mean | `train_joint_clean_patch_l2_linf.py` | `train_joint_clean_linf_l2_patch.py` |

Original `train_trades.py`, `train_mart.py`, `train_derpp_baseline.py`,
`train_gsa_baseline.py`, and `train_joint_pace_aligned.py` remain available.
For Joint-Mean the two entries specify the order of condition forwards in a
joint minibatch, **not** sequential attack stages. They retain Clean → Joint.

## 4. Training commands

The following Linux examples explicitly use Imagenette, ResNet-18, batch 32,
seed 42, all final-test examples, and an independent AutoAttack batch of 4.
On PowerShell, omit `CUDA_VISIBLE_DEVICES=0` or set `$env:CUDA_VISIBLE_DEVICES="0"`
before a single-line Python command. Always give each experiment its own name.

```bash
CUDA_VISIBLE_DEVICES=0 python train_pace_clean_linf_l2_patch.py \
  --dataset imagenette --backbone resnet18 --data-root ./data --batch-size 32 \
  --policy dtr --ablation full --replay-source-mode fg_adaptive_lazy \
  --far-monitor-size 500 --far-history-size 500 \
  --autoattack-samples 0 --autoattack-batch-size 4 --seed 42 --run-name pace_reverse_s42

CUDA_VISIBLE_DEVICES=0 python train_sseat_clean_linf_l2_patch.py \
  --dataset imagenette --backbone resnet18 --data-root ./data --batch-size 32 \
  --autoattack-samples 0 --autoattack-batch-size 4 --seed 42 --run-name sseat_reverse_s42

CUDA_VISIBLE_DEVICES=0 python train_crt_clean_linf_l2_patch.py \
  --dataset imagenette --backbone resnet18 --data-root ./data --batch-size 32 \
  --crt-strategy croce --crt-use-alr --crt-alr-steps 1 \
  --crt-alr-weight-initial 1 --crt-alr-weight-finetune 0.5 \
  --autoattack-samples 0 --autoattack-batch-size 4 --seed 42 --run-name crt_reverse_s42

CUDA_VISIBLE_DEVICES=0 python train_trades_clean_linf_l2_patch.py \
  --dataset imagenette --backbone resnet18 --data-root ./data --batch-size 32 \
  --autoattack-samples 0 --autoattack-batch-size 4 --seed 42 --run-name trades_reverse_s42
```

For MART, DER++, and GSA, replace the TRADES script and run name with the matching
method. TRADES defaults to KL inner attacks; `--trades-inner-loss ce` retains the
legacy alternative. To run the old order, choose the left-column script. For
CIFAR-100 + WRN-34-10 use `--dataset cifar100 --backbone wrn --batch-size 64`;
CIFAR-10 changes only the dataset. Add `--dry-run` to inspect resolved settings
without loading data. Do not pass PACE-only flags to other trainers.

```bash
python train_joint_clean_linf_l2_patch.py --dataset imagenette --backbone resnet18 --data-root ./data --batch-size 32 --autoattack-samples 0 --autoattack-batch-size 4 --seed 42 --run-name joint_reverse_s42
```

### Replay and ablations

PACE supports `--replay-source-mode historical|regenerated|fg_adaptive|fg_adaptive_lazy`
and `--policy dtr|random|uniform|lowest_accuracy`. Lazy caching creates a training
cache only on the first historical-replay use, then reuses it. It is not an
automatic cache-refresh policy.

`--ablation` accepts `full`, `only_cls`, `no_distillation`, `no_contrast`,
`fixed_weights`, `no_queue`, `fixed_temperature`, and `no_dtr`.
Batch launchers: `run_ablations.py`, `run_replay_study.py`, `run_far_source_modes.py`.
They retain the generic entry's old default order; pass
`--task-sequence clean pgd pgd_l2 patch` to launch the reverse order and use a new
output root/run name. Use their `--dry-run` first.

## 5. Selection, resume and evaluation

For a phase of E epochs, W = min(E−1, floor(0.05 E)+1), with a minimum of one
warmup epoch for E>1. Best is eligible when the **one-based epoch > W**.
Default Clean: W=6, first eligible epoch=7. Default attack: W=2, first eligible
epoch=3. A one-epoch diagnostic phase has W=0. Accuracy means include 0%; ties
retain the earlier eligible checkpoint. The scheduler and selector use one
shared function. Warmup epochs still train, validate and save resume state.

This does not make every selection dataset identical: SSEAT/CRT retain full
validation for the current task and the fixed monitor for previous tasks;
PACE and the four additional baselines use full validation for selection.
See [protocol details](docs/protocol.md).

- PACE epoch continuation: repeat the exact command with `--resume --resume-mode epoch`.
  Keep the complete run folder: a not-yet-built adversarial replay cache still needs
  its historical `*_selected.pt`. The resume file embeds panels and already-built
  caches, not every historical model. Clean replay itself needs no attack model.
  Default PACE `--resume` restarts an unfinished task from its preceding boundary.
- SSEAT/CRT: repeat the exact entry and options with `--resume`; restart after the
  last completed stage, not an exact interrupted epoch. SSEAT requires its memory file.
- TRADES/MART/DER++/GSA/Joint: repeat the exact command with `--resume` for epoch resume.
  `--eval-only` requires completed training and the same run identity.
- Keep new and old orders in separate directories. Old selection/ALR protocols
  must not be relabeled as this release. PACE's explicit legacy-import utilities
  remain for compatibility; any such continuation is a mixed-protocol run.
- `--autoattack-samples 0` evaluates the full available test set, whatever its size.
  Positive smaller counts use deterministic class-balanced sampling. This is
  distinct from `--far-monitor-size` and `--tsne-samples`.

## 6. Figures

Add `--plot-final-features --tsne-samples 1000 --tsne-dimensions both` to a training
or completed-run evaluation command. Features come from actual final-test
attacks. If no feature cache exists, evaluation must be run again to collect it;
accuracy numbers alone cannot reconstruct features. Cached figures can be redrawn:

```bash
python plot_final_features.py --run-dir ./runs/your_run --tsne-samples 1000 --tsne-dimensions both --feature-distance l2
```

Outputs include all-condition/per-attack t-SNE, clean reference, distance density
curves and per-attack histograms. Category is encoded by color and attack by
marker. t-SNE uses matched, class-balanced clean IDs across conditions. Additional
boundary copies show a **projected surrogate**, not the original classifier's
true boundary. Distance correctness/error overlays are not decision thresholds.

## 7. Tests and artifacts

```bash
python -B -m pytest -q tests --prune-test-artifacts -p no:cacheprovider
python train_replay_comparison.py --smoke --dataset cifar10 --run-name smoke_check
```

Tests use synthetic data and fake AutoAttack where appropriate; they do not
establish real-dataset accuracy. Three full ResNet multi-stage checkpoint tests
need at least 2 GiB free temporary disk and explicitly skip otherwise. Real
ResNet forward/backward tests are separate. Leave several GiB free for a full run.
GitHub Actions runs the CPU suite; CUDA experiments remain the user's responsibility.
`docs/TEST_REPORT.md` describes this release's local verification.

Run directories contain configs, environment metadata, metrics, checkpoints,
resource records and final-test results. PACE and the baseline families have
different checkpoint filenames; do not rename files to force compatibility.
`export_run_artifacts.py`, `summarize_ablations.py`, and `summarize_replay_study.py`
provide export helpers. Data, runtime caches, private logs, and weights are excluded
by `.gitignore` and are not shipped in this source release.

## Attribution and publication

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for CRT provenance and external
dependencies. No repository-wide license or author/paper metadata was invented
during packaging. Confirm ownership and upstream redistribution terms, then add
an appropriate LICENSE and your final citation before public release. Until
then, stage the repository privately. See the [upload guide](docs/github_upload_zh.md).
