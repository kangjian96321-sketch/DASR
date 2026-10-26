# DASR 持续对抗训练代码

[English](README.md)｜[技术与参数](docs/protocol.md)｜[修改前后对比](docs/changes.md)｜[GitHub 上传操作](docs/github_upload_zh.md)

本合集是 20261025 已验证发布版的 20261026 纯命名更新，包含 DASR、SSEAT-aligned、
CRT-aligned、TRADES、MART、DER++、GSA 和联合训练，以及最终特征可视化。
主入口统一为 `train_DASR.py`，逆序入口为 `train_DASR_clean_linf_l2_patch.py`，
模块统一为 `dasr_aligned` 和 `dasr_components`；导入、批量启动器和测试同步更新。
本次不改变训练算法、参数和默认攻击顺序。见[更名与文件替换清单](docs/RENAME_20261026.md)。
**不含 CAD，也不含独立的 Patch 阶段重训工具。**

## 本版已经统一的内容

- FAR 默认 `monitor=500`、`history=500`，包括命令行和直接调用函数的默认值。
- 所有训练器在各自阶段的 warm-up 结束之后才开始选择 best。
- 对评估条件等权求平均时计入 0%，全 0% 也能选出合法 best；同分保留先出现者。
- 每个方法均提供新旧两种顺序入口。
- CRT 的 L2/L∞ ALR 合入后续官方对齐模块；Patch ALR 保留项目扩展实现。
- 新选模策略进入配置和恢复校验，避免旧实验被直接当成新协议续跑。

**没有改动的内容：**数据划分、原攻击预算、损失主体、DTR/F/G 路由、缓存跨阶段保留、
原对比方法的重放机制。没有合入“每阶段刷新 FAR 历史样本”的分支。
上面列出的 warm-up/计入0%等调整已经包含在 20261025 版，本次没有再次改动。
本次方法标签、默认运行名和部分协议版本字符串随名称更新，旧原生断点可能因身份检查被拒绝。
不要通过修改签名或强行改日志继续训练；正在运行的实验保留原目录，命名版放新目录。

## 1. 安装环境

在解压后的项目根目录运行，二选一：

```bash
# NVIDIA GPU 环境：CUDA 12.1 依赖配方
conda env create -f environment.yml
conda activate dasr

# 或者 CPU 测试环境
conda env create -f environment-cpu.yml
conda activate dasr-cpu
```

已有独立 Python 环境也可使用：

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-autoattack.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CPU 用户将 `cu121` 换为 `cpu`。没有 NVIDIA GPU 时不要安装 GPU 环境配方。
环境 YAML 已包含 AutoAttack 的官方 Git 安装依赖；源码不随包附送。
该 VCS 依赖未固定 commit，请保存安装时解析到的 commit 和 `pip freeze` 输出，
正式比较实验应使用同一 commit，避免上游后续更新造成差异。
YAML 是安装配方，不是完整依赖锁文件；本次实际 CPU 测试环境记录在
`docs/test_environment.json`。没有在本机重新安装 Conda 环境或完成真实 CUDA 训练。

PyTorch 版本配对及其他 CUDA 安装命令参考
[官方安装说明](https://pytorch.org/get-started/previous-versions/)。

## 2. 数据准备

CIFAR-10/100 由 torchvision 下载至 `--data-root`。Imagenette 需要事先解压：

```text
data/imagenette2-160/
  train/类别文件夹/图片.JPEG
  val/类别文件夹/图片.JPEG
```

Imagenette 从 `train/` 中按 `split_seed` 留出 1000 张作为验证集，`val/` 用作最终测试集。
输入尺寸为 160×160。其他支持的数据集默认留出 5000 张验证样本。
Tiny-ImageNet 支持原 `val_annotations.txt`，也支持已经按类别整理的验证目录。
验证集划分本身是固定种子的随机划分；monitor 是从中抽出的类别均衡子集，不能把两者混为一谈。
测试集不参与选模或重放决策。

## 3. 所有方法的新旧两种入口

内部 `pgd` 表示 PGD-L∞，`pgd_l2` 表示 PGD-L2。

| 方法 | 旧顺序：Clean→Patch→L2→L∞ | 新顺序：Clean→L∞→L2→Patch |
|---|---|---|
| DASR | `train_DASR.py` | `train_DASR_clean_linf_l2_patch.py` |
| SSEAT | `train_sseat_dasr_aligned.py` | `train_sseat_clean_linf_l2_patch.py` |
| CRT | `train_crt_dasr_aligned.py` | `train_crt_clean_linf_l2_patch.py` |
| TRADES | `train_trades_clean_patch_l2_linf.py` | `train_trades_clean_linf_l2_patch.py` |
| MART | `train_mart_clean_patch_l2_linf.py` | `train_mart_clean_linf_l2_patch.py` |
| DER++ | `train_derpp_clean_patch_l2_linf.py` | `train_derpp_clean_linf_l2_patch.py` |
| GSA | `train_gsa_clean_patch_l2_linf.py` | `train_gsa_clean_linf_l2_patch.py` |
| 联合训练 | `train_joint_clean_patch_l2_linf.py` | `train_joint_clean_linf_l2_patch.py` |

原 `train_trades.py`、`train_mart.py`、`train_derpp_baseline.py`、`train_gsa_baseline.py`
和 `train_joint_dasr_aligned.py` 仍保留。固定顺序入口不允许再覆盖其顺序。
**联合训练的双入口仅指定同一 minibatch 内条件前向的排列；它仍然是 Clean→Joint，
不是三阶段持续训练。**改变排列可能影响共享 BN 的更新顺序，但不改变均值目标定义。

## 4. Imagenette + ResNet-18 训练命令

以下 Linux 命令统一显式指定 batch32、seed42、全测试集 AA、AA batch4。
依次运行；`CUDA_VISIBLE_DEVICES` 可按实际 GPU 编号调整。

```bash
CUDA_VISIBLE_DEVICES=0 python train_DASR_clean_linf_l2_patch.py \
  --dataset imagenette --backbone resnet18 --data-root ./data --batch-size 32 \
  --policy dtr --ablation full --replay-source-mode fg_adaptive_lazy \
  --far-monitor-size 500 --far-history-size 500 \
  --autoattack-samples 0 --autoattack-batch-size 4 --seed 42 --run-name dasr_reverse_s42

CUDA_VISIBLE_DEVICES=0 python train_sseat_clean_linf_l2_patch.py \
  --dataset imagenette --backbone resnet18 --data-root ./data --batch-size 32 \
  --autoattack-samples 0 --autoattack-batch-size 4 --seed 42 --run-name sseat_reverse_s42

CUDA_VISIBLE_DEVICES=0 python train_crt_clean_linf_l2_patch.py \
  --dataset imagenette --backbone resnet18 --data-root ./data --batch-size 32 \
  --crt-strategy croce --crt-use-alr --crt-alr-steps 1 \
  --crt-alr-weight-initial 1 --crt-alr-weight-finetune 0.5 \
  --autoattack-samples 0 --autoattack-batch-size 4 --seed 42 --run-name crt_reverse_s42

CUDA_VISIBLE_DEVICES=0 python train_trades_clean_linf_l2_patch.py --dataset imagenette --backbone resnet18 --data-root ./data --batch-size 32 --autoattack-samples 0 --autoattack-batch-size 4 --seed 42 --run-name trades_reverse_s42
CUDA_VISIBLE_DEVICES=0 python train_mart_clean_linf_l2_patch.py --dataset imagenette --backbone resnet18 --data-root ./data --batch-size 32 --autoattack-samples 0 --autoattack-batch-size 4 --seed 42 --run-name mart_reverse_s42
CUDA_VISIBLE_DEVICES=0 python train_derpp_clean_linf_l2_patch.py --dataset imagenette --backbone resnet18 --data-root ./data --batch-size 32 --autoattack-samples 0 --autoattack-batch-size 4 --seed 42 --run-name derpp_reverse_s42
CUDA_VISIBLE_DEVICES=0 python train_gsa_clean_linf_l2_patch.py --dataset imagenette --backbone resnet18 --data-root ./data --batch-size 32 --autoattack-samples 0 --autoattack-batch-size 4 --seed 42 --run-name gsa_reverse_s42
CUDA_VISIBLE_DEVICES=0 python train_joint_clean_linf_l2_patch.py --dataset imagenette --backbone resnet18 --data-root ./data --batch-size 32 --autoattack-samples 0 --autoattack-batch-size 4 --seed 42 --run-name joint_reverse_s42
```

旧顺序换成表格左列入口，并给出独立 `run-name`。CIFAR-100 + WRN-34-10 改为
`--dataset cifar100 --backbone wrn --batch-size 64`，CIFAR-10 再替换数据集名。
Windows PowerShell 不支持开头的 Linux 环境变量语法；先执行
`$env:CUDA_VISIBLE_DEVICES="0"`，再运行单行 `python ...` 命令。

所有正式命令均可先加 `--dry-run` 查看配置；不能只改 run-name 就认为顺序已改变。
DASR 的 Imagenette 默认 batch 仍为 64，其他入口通常为 32，所以这里显式指定 32。
不要向 SSEAT/CRT 传 DASR 专有的 `--ablation`、`--far-*`、`--clean-epochs` 等参数。

## 5. 重放选项与消融

| `--replay-source-mode` | 实现 |
|---|---|
| `historical` | 全部历史缓存，预先生成训练缓存 |
| `regenerated` | 全部使用当前模型重新攻击 |
| `fg_adaptive` | F/G 路由，预先生成训练缓存 |
| `fg_adaptive_lazy` | F/G 路由，首次需要历史重放才生成训练缓存 |

惰性缓存减少未被使用缓存的提前生成，不等于每个 batch 都重建，也不等于自动刷新过期缓存。
`monitor=500` 是每个任务用于监测的固定验证样本规模；`history=500` 是每个任务的训练缓存规模，
来自训练集，不是将验证样本用于训练。默认缓存精度是 float32，`uint8` 可选但会改变量化精度。

任务选择 `--policy` 支持 `dtr/random/uniform/lowest_accuracy`。
消融支持 `full/only_cls/no_distillation/no_contrast/fixed_weights/no_queue/fixed_temperature/no_dtr`。
批量运行脚本 `run_ablations.py`、`run_replay_study.py`、`run_far_source_modes.py` 默认仍使用旧顺序；
逆序需显式追加 `--task-sequence clean pgd pgd_l2 patch`，并使用新的输出根目录。

## 6. best、恢复与最终测试

实际选模条件是 `epoch + 1 > warmup_epochs`，其中 epoch 为从 0 开始的索引。
默认 100 轮 Clean 的 warm-up 为 6 轮，第 7 轮开始选；30 轮攻击为 2 轮，第 3 轮开始选。
warm-up 期间仍进行训练、验证及断点保存，但不把这些模型当作 best。
单轮诊断训练没有 warm-up，可选择唯一一轮；正式默认日程不受该边界处理影响。

所有方法计入 0%，但本次没有把验证样本范围也统一：SSEAT/CRT 仍然当前任务全验证、
历史任务固定子集选模；DASR 和四个额外基线使用完整验证集。详见 `docs/protocol.md`。

- DASR 精确到 epoch 续训：原完整命令追加 `--resume --resume-mode epoch`。
  请保留完整运行目录：尚未生成的历史攻击缓存仍需要对应的 `*_selected.pt`。
  resume 文件包含监测面板和已生成缓存，不包含所有历史模型；Clean 重放本身不依赖攻击模型。
- DASR 默认仅 `--resume`：最新未完成任务从阶段起点重新训练。
- SSEAT/CRT：同入口、同设置追加 `--resume`，从上一个已完成阶段继续；不是精确到 epoch。
  SSEAT 必须保留 `adr_memory.pt`，不能只复制权重。
- TRADES/MART/DER++/GSA/Joint：追加 `--resume` 恢复最近完整 epoch；训练结束后可 `--eval-only`。
- 新旧序列、不同 batch、不同选模协议、不同 CRT ALR 配置不能在同一目录强行续训。
- DASR 保留显式旧 checkpoint 导入能力，但这类继续训练需标记为混合协议，不能作为本版从头训练结果。

`--autoattack-samples 0` 表示完整测试集，不固定为 10,000 张；具体数量取决于数据集。
正数且小于测试集时采用类别均衡子集。`--autoattack-batch-size` 只影响 AA 的计算批量。
最终常规条件包括 Clean、Patch、CW-L2、PGD-L2、PGD-L∞，另测 AA-L2/L∞。

## 7. 散点图和特征距离图

正式运行或补做最终评估时加：

```bash
--plot-final-features --tsne-samples 1000 --tsne-dimensions both
```

仅重绘已有特征缓存，不再生成攻击：

```bash
python plot_final_features.py --run-dir ./runs/你的实验目录 --tsne-samples 1000 --tsne-dimensions both --feature-distance l2
```

输出总图、Clean 图、Clean 与各攻击的分图、距离密度总曲线和每攻击直方图。
类别用颜色区分、攻击用符号区分；t-SNE 使用各条件共有、类别均衡、源样本一一对应的子集。
边界副本是投影空间的近似预测分界，不是原模型真实分类边界；距离图中的正确/错误分布也不是阈值。
只有准确率 JSON、没有特征缓存时，必须加载模型重新评估，无法从准确率反推出散点图。

## 8. 测试与上传

```bash
python -B -m pytest -q tests --prune-test-artifacts -p no:cacheprovider
python train_DASR.py --smoke --dataset cifar10 --run-name smoke_check
```

三项完整 ResNet 多阶段 checkpoint 测试需要至少 2 GiB 临时磁盘空间，不足会明确跳过。
其余测试含合成数据训练、重放、恢复、真实网络前向/反向、模拟 AA 及绘图，
不等于重新完成 Imagenette/CIFAR 的正式 GPU 实验。
`--prune-test-artifacts` 仅清理 pytest 生成的函数级临时文件，失败调试时可不加。

源代码包不含数据集、权重、真实实验日志或本机绝对路径。`.gitignore` 也排除了这些运行产物。
实际上传操作见 [GitHub 上传说明](docs/github_upload_zh.md)。上传前需自行确认作者、论文信息、
代码权属和第三方许可；本包没有擅自添加 MIT 等仓库级许可证。
