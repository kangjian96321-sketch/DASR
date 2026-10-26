# Implementation and parameters / 实现与参数

This document describes this source snapshot, not an assertion of full official
baseline reproduction. 中文与英文 README 中的命令对应本文件列出的实现。

## DASR

- DTR selects a previous task using its accuracy decrease relative to the preceding
  stage's selected model. The default gate is strictly greater than 1 percentage
  point. Clean is included unless `--exclude-clean`; stage epoch 0 does not replay.
- F/G uses fixed historical examples from the original best checkpoint of each
  task. The monitor is eager, while training caches follow the chosen replay mode.
  **The cache generation model is not refreshed at each new stage.**
- With baseline accuracy A, current accuracy B on the fixed historical examples,
  and current fresh-attack accuracy C, the signed decomposition is F=A−B, G=B−C,
  F+G=A−C. The implementation logs `a_ss`, `a_ts`, `a_tt`. The preceding-stage
  baseline update is retained from the source branch; this is not a causal
  decomposition nor a guarantee that historical examples remain optimal attacks.
- Given threshold d (default 0 pp), p(history)=max(F−d,0)/[max(F−d,0)+max(G−d,0)].
  If the denominator is zero, p(history)=0. The regenerated probability is 1−p.
  Signed values are retained in logs; clipping is only for routing probabilities.
- Monitor and train-cache defaults are both 500 per task. Monitor IDs are drawn
  from validation; replay-cache IDs come from training. Fresh monitor evaluation
  is reused from full validation; this does not make monitoring entirely free.
- LACD matches student attacked queries to paired clean keys from the preceding
  stage's teacher. Positive temperatures vary over 0.05–0.15; negative temperature
  is 0.1. Current/replay temperature normalization is separate; negatives remain
  pooled. The class-balanced feature queue has capacity 256, clears at task
  boundaries, and enqueues current clean keys once per minibatch.
- Gradient cosine smoothing: 0.8; adaptive weight floor: 0.2. Base feature weights
  are Patch/L2/Linf = 1/1/3. Clean initialization trains CE only.
- Eight ablation presets and four task selectors remain available. No new routing
  policy, cache refresh, or adaptive attack-strength schedule is introduced here.

## Shared training settings

| Setting | Value |
|---|---|
| Network | WRN-34-10 or ResNet-18, no pretrained weights |
| Inputs | CIFAR 32; Tiny-ImageNet 64; Imagenette 160 |
| Clean / attack epochs | 100 / 30 each |
| Clean / attack peak LR | 0.1 / 0.01 |
| Optimizer | SGD, momentum 0.9, weight decay 0.0005 |
| LR schedule | warmup ratio .05, start factor .1; cosine minimum .001 × phase peak LR |
| EMA | .999, trainable parameters only; current BN buffers retained |
| Seed / split / evaluation seed | 42 / 42 / 42 |
| Validation size | Imagenette 1000; other supported datasets 5000 |
| DASR batch defaults | CIFAR 64, Tiny 16, Imagenette 64 |
| Other continual trainers' batch defaults | CIFAR 64, Tiny 16, Imagenette 32 |
| Joint defaults | Imagenette, ResNet-18, batch32; Clean100 + Joint90 |

Explicit batch32 is used throughout the Imagenette README examples. This release
does not silently replace DASR's previous batch64 default.

## Attacks in pixel space [0,1]

| Parameter | CIFAR | Tiny-ImageNet | Imagenette |
|---|---:|---:|---:|
| Linf epsilon | 8/255 | 8/255 | 8/255 |
| Linf step | 2/255 | 2/255 | 2/255 |
| L2 epsilon | .5 | 1.0 | 2.5 |
| L2 step | .1 | .2 | .5 |
| Patch side | 8 | 16 | 40 |
| Ordinary train / eval steps | 10 / 20 | 10 / 20 | 10 / 20 |
| CW-L2 eval steps | 30 | 30 | 30 |

Patch uses the existing pixel-replacement constraint and sign updates of 0.1.
These budgets are inherited experimental settings, not dimension-normalized
equivalent attack strengths. DASR resolves them in `train_DASR.main`;
shared baselines resolve them in `common.config_from_args`. Four-baseline and
joint CLIs also expose explicit epsilon/step overrides. The DASR/SSEAT/CRT entries
do not expose every such override: inspect `--help`, do not assume flags transfer.

## Method-specific defaults

| Method | Mechanism / parameters |
|---|---|
| SSEAT | ADR + CRS; memory CIFAR10/Imagenette1000, CIFAR1002000, Tiny4000; replay batch8; 12 uncertainty views; clean reserve .1; temperature1; CRS and replay weights1 |
| CRT | Croce attack sampling; Lp ALR1 step; weight1 for first adversarial stage, .5 thereafter; clean has no ALR |
| TRADES | KL inner objective by default, CE option retained; beta Patch/L2/Linf=1/1/6 |
| MART | boosted adversarial CE + misclassification-aware KL; beta1/1/6 |
| DER++ | exposure reservoir5120; fixed adversarial image, label and pre-update logits; alpha=.5, beta=.5 |
| GSA | exposure reservoir5120 clean views; CURRENT-stage re-attack; gradient projection plus reference gradient |
| Joint | equal mean CE over selected conditions, shared BN, no teacher/replay/LACD/ALR |

DER++/GSA replay batch defaults to training batch; their memory remains the final
exposure reservoir when the model rewinds to a stage best. These are this
project's variants; GSA is not claimed to reproduce an unidentified upstream paper.

## Checkpoint selection

`dasr_aligned/selection.py` defines the mean, warmup length and eligibility for
every trainer. Means include exactly the specified conditions, including zero;
missing/extra conditions, NaN, infinity and values outside [0,100] are rejected.
Default eligibility: Clean epoch7+, attack epoch3+; one-epoch diagnostics have
zero warmup. Equal scores do not replace the existing eligible best.

The **mean formula and time gate are unified, not every validation sample**:

| Trainer | Selection set |
|---|---|
| DASR, TRADES, MART, DER++, GSA | full validation for every seen task |
| SSEAT, CRT | current task full validation; historical tasks fixed monitor (500 by default) |
| Joint | full validation for phase conditions |

SSEAT/CRT still log full-validation historical accuracies separately. This known
difference must be disclosed when presenting aligned comparisons. It was not
silently changed by the request to count zero accuracies.

## CRT ALR provenance and scope

`dasr_aligned/crt_lp_alr.py` is the device-portable later alignment of
[official L2Reg(var_reg=False)](https://github.com/inspire-group/continual_robust_training/blob/5a6932430219d531aab5736de94292b149ff73ae/train/reg_utils.py),
pinned to commit `5a6932430219d531aab5736de94292b149ff73ae`.
It preserves training mode, per-step clean forwards, CPU float32 random starts,
Lp projections, zero-gradient handling and adversarial-first outer forwards.
Both clean/adversarial outer branches differentiate model parameters. It avoids
discarded inner weight gradients and redundant upstream double calls.

This is single-call algorithm alignment, not full-training trajectory parity.
Patch ALR is unchanged and is **not** called an official CRT Patch algorithm.
StAdv/ReColor/other attack branches are not added. Optional parity checker:

```bash
python check_crt_lp_alr.py --crt-repo ./external/continual_robust_training --device cpu
```

Use a trusted checkout containing the pinned commit, or `--reference-file` with
the exact upstream file (SHA256 enforced). The external repo is not distributed.

## Evaluation and reporting

Ordinary final-test conditions: Clean, Patch, CW-L2, PGD-L2, PGD-Linf; plus AA-L2
and AA-Linf. Positive AA subsets are class balanced; full-test mode preserves all
test examples. DASR/SSEAT/CRT default AA sample count1000; four-baseline and joint
CLIs default0. All README training examples explicitly request0 for comparability.

Feature caches store actual paired final-test features on CPU. t-SNE defaults:
1000 balanced source IDs, perplexity30, iterations1000, both2D/3D; distance bins60,
metric raw L2. Matching IDs, not independently sampled adversarial classes, are
used for clean/attack comparison. Surrogate boundaries must not be interpreted
as the true high-dimensional model boundary or as evidence of causal stability.
