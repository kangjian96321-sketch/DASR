# Rename-release verification / 命名版验证

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
