# Provenance and third-party components

This release is assembled from the supplied DASR research project and its later
CRT Lp ALR alignment. It does not grant a new repository-wide license.

- CRT algorithm reference: [inspire-group/continual_robust_training](https://github.com/inspire-group/continual_robust_training),
  commit `5a6932430219d531aab5736de94292b149ff73ae`, `train/reg_utils.py:L2Reg`.
  `dasr_aligned/crt_lp_alr.py` is a local device-portable adaptation of that algorithm.
  The parity checker reads an external trusted reference with SHA256
  `87ba0a7f348ab802f03fd31156abc860a0809147bb9b1fc0dc1bbd56db9a937d`.
  No full upstream checkout is bundled. A redistribution license for that exact
  upstream snapshot was not established during packaging; confirm its terms or
  obtain permission before publicly distributing derivative code.
- [AutoAttack](https://github.com/fra31/auto-attack) is installed separately under
  its own upstream terms. It is not vendored here.
- PyTorch, torchvision, NumPy, Matplotlib, scikit-learn and Pillow are dependencies,
  not relabeled as original project code. Their original licenses apply.
- Dataset access and redistribution are governed by the dataset providers; no
  dataset files or trained weights are included.
- SSEAT, CRT, TRADES, MART, DER++ and GSA names identify method references or the
  documented aligned variants, not ownership of their original research.

Before public release, the maintainer should confirm rights to all supplied
source, preserve any applicable upstream notices, select a compatible license
for code they own, and add the final paper citation/author information. A public
GitHub repository without a license is not automatically an open-source license.
Until those decisions are made, private staging is the intended upload workflow.
