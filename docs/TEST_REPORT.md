# Release verification / 发布验证

Date: 2026-10-25. Target: `DASR_PACE_GitHub_20261025`.
Runtime: Windows, Python 3.12.14, PyTorch 2.4.1+cpu.
Exact observed package versions: [test_environment.json](test_environment.json).

## Final regression

**299 passed, 3 skipped, 0 failures, 0 errors** in 126.54 seconds.
There were 62 warnings: existing PyTorch scheduler epoch-argument deprecation and
Matplotlib/pyparsing deprecations; these are not suppressed in the runner.

```bash
python -B -m pytest -q tests --prune-test-artifacts -p no:cacheprovider
```

The final local run also set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`, used a
separate pytest temporary directory, and exported JUnit XML outside the package.
Earlier diagnostic runs revealed and led to fixes for warmup recovery, Clean
lazy-cache recovery, and CRT CLI validation. Their partial results are not used
as evidence that the final suite passed.

Covered:

- New and old sequence entry points and explicit configuration propagation.
- monitor/history defaults 500/500; balanced sampling and full-validation reuse.
- Scheduler-aligned best eligibility, all-zero means, ties, invalid values.
- Warmup interruption and resumed selection without manufacturing a warmup best.
- Synthetic-data training and checkpoint recovery for the supported methods.
- Fixed historical/regenerated/F-G/eager/lazy replay routing and cache behavior.
- Real backbone shape/forward/backward checks, including ResNet inputs.
- AutoAttack batching and balanced subset selection using controlled substitutes.
- Feature plotting, shared 2D/3D coordinates and paired input indexing.
- CRT Lp ALR dispatch, gradients, train/eval modes and BatchNorm effects.

Three tests explicitly skipped because less than 2 GiB free temporary disk was
available:

1. ResNet SSEAT four-stage training + resume.
2. ResNet CRT four-stage training + resume.
3. ResNet PACE F/G-lazy four-stage training + queue + resume.

These are **not counted as passed**. Run the same suite with several GiB free to
exercise them; GitHub Actions is configured but has not been run remotely here.

## CRT reference parity

**60 / 60 passed**, maximum absolute difference **0** in this CPU float32 run.
Details: [crt_alr_parity.json](crt_alr_parity.json).

- Reference: official `L2Reg(var_reg=False)` at commit
  `5a6932430219d531aab5736de94292b149ff73ae`.
- Reference SHA256 enforced by `check_crt_lp_alr.py`.
- 2 norms × 5 step counts (1,2,3,4,10) × 2 model modes × 3 test models.
- Compared generated perturbations, scalar loss, parameter gradients, BN state.
- The reference is adjusted only to replace hard-coded CUDA placement with the
  requested device. It is not bundled with this source package.

This verifies individual regularizer calls, **not full official CRT training
trajectory parity**. Patch regularization remains the project's extension.

## Scope and limits / 测试边界

本次未重新运行真实 CIFAR/Imagenette 全程 GPU 实验、真实完整 AutoAttack，
也未在干净系统重新求解/安装 Conda 环境。因此不能据此承诺论文准确率、GPU 峰值显存、
训练时间或所有平台的安装成功。AutoAttack 安装依赖未固定 commit，正式实验需记录并固定。

本版选模规则会改变后续阶段起点，旧实验结果不能直接作为本版结果。
SSEAT/CRT 保留历史任务子集选模；这项验证范围差异并未因为统一均值和 warmup 而消失。

The distribution contains source, docs, environment recipes and tests only.
Python syntax and Markdown relative links are audited before packaging; the ZIP
is reopened for CRC and per-file SHA256 verification. The generated
`release_manifest.json` lists source changes and hashes, excluding its own hash.
