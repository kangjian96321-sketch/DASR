# 完整验证集选模更新（2026-11-02）

## 基础版本与范围

基于本次核对的 GitHub 命名版 `DASR_GitHub_20261026`。
本地最新下载的 `DASR-main.zip` 与其 90 个文件逐个内容相同。
未混入共享首任务等其他实验分支。

本次仅统一 best 的验证样本范围并复用验证结果，保留训练损失、重放机制、
攻击强度、数据划分、warm-up 门槛、EMA 规则及 0% 计入平均的规则。
两个攻击顺序的入口共用这些实现，均生效。

## 最终规则

设截至当前阶段已见任务集合为 S，V 是当前数据划分的完整验证集。
每个验证轮次在同一个待选模型上，为每个任务使用对应攻击及原验证攻击配置，
计算该模型在 V 上的准确率；对 S 中所有准确率等权求平均。
warm-up 结束后，平均值严格高于此前最佳值才替换 best；同分保留先出现的模型。
Clean 也是一个评估条件，成为历史任务后仍使用完整 V。

| 方法 | 修改前 | 修改后 |
|---|---|---|
| DASR、TRADES、MART、DER++、GSA | 各已见任务完整验证集 | 保持不变 |
| SSEAT、CRT | 当前任务全量，历史任务固定 500 子集选模；另算全量写日志 | 各已见任务完整验证集，同一结果用于选模和日志 |
| 联合训练 | 当前阶段所有训练条件的完整验证集 | 保持不变 |

默认 Imagenette 验证集为 1000 张，其他支持的数据集为 5000 张；若自定义
`--val-size`，则使用该划分的全部样本。验证集不是最终测试集。

`far_monitor_size=500`、`far_history_size=500` 不变：它们控制监测面板和历史训练缓存，
不等同于 best 选择使用的验证集大小。DTR/F/G 路由未改动。
保留 `selection_val` 数据加载器供已有监测路径使用，不再用于 SSEAT/CRT 选 best。
`max_eval_batches`/smoke 等显式诊断限制仍然有效；正式实验须关闭验证截断。

## 必须替换的四个运行文件

按项目根目录的相对路径覆盖：

1. `dasr_aligned/common.py`：历史任务使用 `loaders.val`。
2. `dasr_aligned/sseat.py`：复用全量结果，删除历史任务重复验证。
3. `dasr_aligned/crt.py`：同上。
4. `dasr_aligned/selection.py`：更新选模协议标识，防止混用旧断点。

入口脚本不必替换。替换包同时提供更新的 README、协议文档和新增回归测试。
四个运行文件与 20261028 阶段缓存刷新版的对应基础文件也一致；
若使用共享 Clean/共享首任务分支，其实现不同，请勿直接覆盖这四个完整文件。

## 修改前后

原 `common.evaluate_for_selection`：

```python
result.update(evaluate_seen(model, loaders.selection_val, historical, cfg, device))
```

新版：

```python
result.update(evaluate_seen(model, loaders.val, historical, cfg, device))
```

原 SSEAT/CRT 在获得选模结果后，另算历史任务的完整验证集用于日志：

```python
historical = [name for name in seen if name != current_task]
validation = {current_task: selection[current_task]}
if historical:
    validation.update(evaluate_seen(model, loaders.val, historical, cfg, device))
mean_accuracy = sum(validation.values()) / len(validation)
selection_mean = mean_selection(selection, seen)
```

新版复用完整验证结果：

```python
validation = dict(selection)
selection_mean = mean_selection(selection, seen)
mean_accuracy = selection_mean
```

SSEAT 的局部当前任务变量名为 `task`，CRT 为 `current_task`。
正式日志中的 `selection_<任务>` 与 `val_<任务>` 相等，
`selection_mean` 与 `val_mean` 相等。

原协议：`post_warmup_mean_all_seen_including_zero_v1_20261025`

新协议：`post_warmup_mean_all_seen_including_zero_full_validation_v2_20261102`

## 计算开销与实验迁移

旧 SSEAT/CRT 每个验证轮次已经为历史任务计算了“子集一次＋全量一次”。
新版仅保留全量一次，省去重复的子集攻击与前向；不增加验证集驻留 GPU 的规模。
仍按原 batch 分批评估。实际耗时取决于攻击与设备，未进行 GPU 时间测量。

请在独立代码目录/新 `--run-name` 中启动新版实验。不要覆盖正在运行的代码。
全局选模协议标识进入所有方法的配置与恢复检查，因此旧版原生断点不能直接
作为本版 `--resume` 使用，包括本来已经采用完整验证集的其他方法。
这不意味着旧模型权重或旧实验结果失效；只是新旧协议分别记录。
不应修改旧 config 中的版本号来绕过检查。

旧 SSEAT/CRT 的 best 不会因更换文件自动变成全量选出的 best；若需严格的新协议
对比，应重新训练。选中阶段 best 可能变化，后续教师、记忆及训练轨迹也可能变化。

## 验证

新增测试覆盖完整样本计数、历史 Clean、当前结果复用、两种序列下 SSEAT/CRT
选中全量最优模型、验证不重复、日志一致、warm-up 排除以及新协议续训/旧协议拒绝。
具体运行结果见 `TEST_REPORT.md`。合成 CPU 回归不代表完整 GPU 精度实验。
