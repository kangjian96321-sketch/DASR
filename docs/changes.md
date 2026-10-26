# 本次修改前后对比

基础：20261022 Clean→PGD-L∞→PGD-L2→Patch 项目。只修改发布副本，原项目不动。

| 内容 | 修改前 | 本发布版 |
|---|---|---|
| CAD | 入口、模型、配置、测试 | 从本合集移除 |
| Patch 重训工具 | 上次草稿中曾复制 | 不纳入发布 |
| FAR 默认 monitor/history | 1000/2000 | 500/500；CLI、函数、缓存类一致 |
| 四基线顺序 | 依赖模块全局 TASKS，原入口旧顺序 | 使用 cfg.task_sequence，双入口，不修改全局变量 |
| Joint 双入口 | 单通用入口 | 两种条件前向顺序，训练仍 Clean→Joint |
| PACE best | epoch>5，即实际第7轮起 | epoch+1>实际 warmup_epochs |
| SSEAT/CRT/四基线/Joint best | 第一轮就可选 | 同一 warmup 判定函数 |
| SSEAT/CRT 均值 | 排除0% | 所有已见条件计入0%，同时检查无效值 |
| warmup 中断恢复 | PACE 可能将无 best 状态补成 best | warmup 期不生成或补造 best |
| Clean 惰性缓存恢复 | 可能不必要地要求独立 Clean best 文件 | 恒等变换直接复用固定干净面板；其他攻击仍要求保留生成模型 |
| Joint warmup best 文件 | 原逻辑每轮写 best_snapshot | 尚无 best 时不写 None checkpoint |
| CRT Lp ALR | eval 模式、缓存 clean reference 等旧实现 | 后续官方 L2Reg 对齐模块 |
| CRT Patch ALR | 项目自定义扩展 | 保留，不称为官方 Patch 算法 |
| 恢复保护 | 原协议标识/部分字段检查 | 新 selection_policy + CRT ALR 配置校验 |
| CRT 最终评估显存 | 阶段 checkpoint/优化器可能存活 | CPU 加载 checkpoint，及时释放阶段对象 |
| 说明与环境 | 分散且部分过期 | 中英 README、参数表、环境 YAML、CPU CI |

核心判定：

```python
# Before (PACE, zero-based epoch)
if epoch > 5 and score > best_score:
    ...

# After (all trainers; shared implementation)
if best_eligible(cfg, task, epoch) and score > best_score:
    ...

# best_eligible is equivalent to:
epoch + 1 > warmup_epochs(cfg, task)
```

```python
# Before (SSEAT / CRT)
valid_selection = [value for value in selection.values() if value > 0]

# After
selection_mean = mean_selection(selection, seen)
```

`warmup_epochs` 与实际 scheduler 共用：默认 E>1 时
`max(1, min(E-1, int(ratio*E)+1))`。E=1 或显式 ratio=0 时没有 warmup，
scheduler 同步走无 warmup 分支，不只是放宽 best 判定。

未改变：历史缓存仍跨阶段保留；DTR/F/G 阶段准确率基准仍采用上一阶段 best；
各方法原始 replay/loss 主体、预算、数据划分不变。SSEAT/CRT 的历史任务选模子集仍保留。
四基线 monitor 默认也设为500，避免 Imagenette 验证集1000张时严格均衡配额不足。

本次调整了旧测试中与当前实际 batch、协议指纹、固定 epoch>5 假设不符的断言；
新增 warmup、全零选模、双顺序、CRT ALR、恢复保护测试。资源不足的完整 ResNet
checkpoint 用例明确跳过，不伪报通过。具体测试结果见 TEST_REPORT.md。
