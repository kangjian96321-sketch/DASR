# Full-validation selection verification / 完整验证集选模验证

Date: 2026-11-02. Target: `DASR_full_validation_20261102`.
Base: verified `DASR_GitHub_20261026`; 90 source files before this update.
The newly downloaded `DASR-main.zip` matched all 90 base files byte for byte.

## Current regression

**323 passed, 0 skipped, 0 failures, 0 errors**, 145.91 seconds.
67 warnings concern existing scheduler and Matplotlib/pyparsing deprecations.

```bash
python -B -m pytest -q tests --prune-test-artifacts -p no:cacheprovider
```

The actual command also used a dedicated test temporary directory, a JUnit
report outside the source package, and OMP_NUM_THREADS=1/MKL_NUM_THREADS=1.
Runtime: Windows, Python 3.12.14, PyTorch 2.4.1+cpu. This is a CPU regression,
not a new real-dataset GPU experiment. No network dataset download was required.

All inherited tests passed. Unlike the 20261026 test run, disk capacity was
sufficient for the three full ResNet-18 multi-stage training/resume checks;
they executed and passed rather than being skipped.

## New selection checks

Nine new cases passed, including:

- Every validation example, including the partial final batch, participates
  in current and historical task selection, including historical Clean.
- Optional reuse of a known full-validation current-task accuracy does not
  repeat the current evaluation.
- Both attack orders and both native SSEAT/CRT trainers select the full-set
  optimum even when the small monitoring subset prefers another epoch.
- Warmup maxima remain ineligible; each condition is evaluated once per
  validation round, and selection/logging values match exactly.
- Completed-stage resumes work under the new protocol without retraining.
- An old selection-policy config is rejected before its metadata is changed.

The ranking tests run native training, EMA evaluation context, checkpoint
saving and resume, with controlled accuracy outputs to make the choice
unambiguous. Separate sample-count tests execute the real accuracy evaluator
with a tiny model and identity attack stub. These test scopes are not claims
of robust accuracy under real attacks.

## Current change scope and limits

Only four existing runtime Python files change: common.py, sseat.py, crt.py
and selection.py under dasr_aligned/. One test module is added. DASR training,
FAR routing, the four-baseline trainer, Joint and the CRT Lp ALR implementation
are byte-identical to the base release. README and protocol notes are updated.
Source syntax, changed-file scope and packaged file bytes are checked by the
packager. `release_manifest.json` records SHA256 hashes excluding itself.

The subsequent README-only review corrected obsolete rename descriptions and
the selector-study launcher's supported arguments. All 9 distinct README
training examples and 21 documented entry points passed CLI dry runs;
16 local Markdown links and the 2 sequence-forwarding batch launchers were
checked. No training implementation changed during this documentation review,
so the 323-test result above remains the relevant runtime verification.

未重新进行 Imagenette/CIFAR 正式 GPU 训练、完整 AutoAttack 或干净 Conda 安装验证。
未测量真实 GPU 验证耗时；节省子集评估的结论来自调用路径和计数测试。
这次改变选模依据，不能保证 SSEAT/CRT 的 best、后续训练轨迹或最终精度与旧版相同。
请使用新结果目录，保留旧实验；全局选模协议版本更新后，旧原生断点不能直接续训。
本次没有自动修改 GitHub。

---

# Archived 20261026 report / 以下为旧命名版的历史验证记录

The following results describe the base release only. They are retained for
provenance and do not replace the current verification above.

Date: 2026-10-26. Target: `DASR_GitHub_20261026`.
Base: the verified 86-file release dated 2026-10-25.
Observed local CPU environment is recorded in [test_environment.json](test_environment.json).

## Regression

**311 passed, 3 skipped, 0 failures, 0 errors**, 121.27 seconds.
62 warnings concern existing scheduler / Matplotlib / pyparsing deprecations.

```bash
python -B -m pytest -q tests --prune-test-artifacts -p no:cacheprovider
```

The local command additionally used a dedicated pytest temporary directory and
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. JUnit results are retained outside the
source package, without bundling machine-specific temporary paths.

Coverage includes every retained regression from the base release, plus new
DASR CLI naming, old/new sequence defaults, module imports and absence of old
entry-point files. Existing cases cover routing, sampling, validation reuse,
warmup/zero selection, checkpoint recovery, backbones and plotting.

Three tests explicitly skipped because less than 2 GiB free temporary space was
available: full ResNet multi-stage SSEAT, CRT, and DASR F/G-lazy training+resume.
They are not reported as passed. Real ResNet forward/backward checks are separate.
No GitHub Actions run was triggered by this local work.

## Before/after numerical comparison

**2 attack orders, 8 stages, all exact comparisons passed.**
Details: [rename_parity.json](rename_parity.json).

Both original and renamed code ran in independent Python processes, seed42,
CPU, synthetic CIFAR-shaped inputs, WRN-10-1, 3 epochs per stage and one training/
evaluation batch per epoch. Uniform task selection and F/G-lazy replay were used.

Compared tensors and values:

- Selected model parameters/buffers, chosen epoch, score and full validation.
- Resume model, EMA, optimizer, scheduler, queue, F/G metrics, reference and
  observed accuracies, best score, running gradient similarity.
- Per-epoch numerical logs after excluding names and profiling/time fields.

This is a bounded deterministic rename check, not a new real-dataset experiment
or a guarantee of bitwise equality across hardware/software environments.
Method names and protocol fingerprints intentionally differ.

## Static integrity and retained ALR evidence

All 67 inherited Python files retain exactly the same AST after applying only
the documented naming substitution. One new naming-test module is added.
The CRT Lp ALR implementation is byte-identical to the base version under its
new path. [crt_alr_parity.json](crt_alr_parity.json) retains the **20261025**
60-case upstream parity evidence; it is not presented as a fresh GPU evaluation.

The release is audited for valid Python syntax, unresolved old import/entry
names, local Markdown links and excluded runtime artifacts. Source files and
ZIP contents receive SHA256 verification. See `release_manifest.json`.

## Limits / 限制

未重新进行正式 Imagenette/CIFAR GPU 训练、完整 AutoAttack 或干净 Conda 安装验证。
本次不改变算法和数值参数，但更名会影响运行标签、部分配置指纹和旧原生断点识别。
不要覆盖正在运行的服务器代码；旧实验使用原版本，新版使用独立目录。
本次没有自动修改 GitHub，也没有建立新的第三方源码再发布许可。
