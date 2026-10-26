"""Active queue/loss functions extracted from the supplied nested DASR code.

These functions retain its concatenated-batch negative multiplicity and detached
adaptive temperature. Replay view groups can normalize positive temperatures
separately while retaining one joint negative pool.
"""
import torch
from torch import nn
from torch.nn import functional as F

USE_HARD_MINING = False
LACD_TOP_K = 64
USE_FOCAL_WEIGHTING = False
FOCAL_GAMMA = 2.0
FOCAL_MARGIN = 0.0
USE_SOFTPLUS_MARGIN = False
SOFTPLUS_SIM_TARGET = 0.95
SOFTPLUS_BETA = 60.0
USE_VARIANCE_REG = False
VAR_GAMMA = 1.0
VAR_WEIGHT = 0.5
TEMP_POS_MIN = 0.05
TEMP_POS_MAX = 0.15
LACD_TEMP_NEG = 0.1

class PureFeatureDistillationLoss(nn.Module):
    """Paired clean-target cosine distance, with no temperature or negatives."""
    def __init__(self):
        super().__init__()
        self.epoch_distances = []

    def forward(self, z_adv_s, z_clean_t, labels=None, queue_features=None,
                queue_labels=None, instance_weights=None):
        queries = F.normalize(z_adv_s, dim=1, p=2, eps=1e-8)
        keys = F.normalize(z_clean_t.detach(), dim=1, p=2, eps=1e-8)
        distances = 1.0 - F.cosine_similarity(queries, keys, dim=1)
        return distances.mean() if instance_weights is None else (distances * instance_weights).mean()

class TeacherFeatureQueue:
    """Class-balanced FIFO queue containing labelled clean teacher features.

    The total capacity is divided across classes as evenly as possible. Each
    class owns an independent ring-buffer segment, so a frequent class cannot
    evict features belonging to a rare class. Class capacities differ by at
    most one when ``capacity`` is not divisible by ``num_classes``.
    """

    def __init__(self, capacity=4096, num_classes=10, device='cuda'):
        if capacity < num_classes:
            raise ValueError(
                f"TeacherFeatureQueue capacity ({capacity}) must be at least "
                f"the number of classes ({num_classes})."
            )
        self.capacity = int(capacity)
        self.num_classes = int(num_classes)
        self.device = device
        self.features = None
        self.labels = None

        base_quota, remainder = divmod(self.capacity, self.num_classes)
        self.class_capacities = [
            base_quota + (1 if class_id < remainder else 0)
            for class_id in range(self.num_classes)
        ]
        self.class_offsets = []
        offset = 0
        for class_capacity in self.class_capacities:
            self.class_offsets.append(offset)
            offset += class_capacity
        self.class_ptrs = [0 for _ in range(self.num_classes)]
        self.class_counts = [0 for _ in range(self.num_classes)]
        self.is_ready = False

    @torch.no_grad()
    def enqueue(self, keys, labels):
        # 延迟初始化：根据真实特征维度创建张量
        if self.features is None:
            feat_dim = keys.shape[1]
            self.features = torch.zeros(self.capacity, feat_dim, device=self.device)
            self.labels = torch.full((self.capacity,), -1, dtype=torch.long, device=self.device)

        # L2 归一化后入队
        keys = F.normalize(keys, dim=1, p=2)
        labels = labels.to(device=self.device, dtype=torch.long)

        for class_id in labels.unique(sorted=True).tolist():
            if class_id < 0 or class_id >= self.num_classes:
                raise ValueError(
                    f"Queue label {class_id} is outside [0, {self.num_classes - 1}]."
                )

            class_keys = keys[labels == class_id]
            class_capacity = self.class_capacities[class_id]
            class_offset = self.class_offsets[class_id]

            # When one mini-batch contains more than the class quota, only the
            # most recent quota is retained, exactly as in a FIFO ring buffer.
            if class_keys.shape[0] >= class_capacity:
                class_keys = class_keys[-class_capacity:]
                start = class_offset
                end = class_offset + class_capacity
                self.features[start:end] = class_keys
                self.labels[start:end] = class_id
                self.class_ptrs[class_id] = 0
                self.class_counts[class_id] = class_capacity
                continue

            for key in class_keys:
                local_ptr = self.class_ptrs[class_id]
                slot = class_offset + local_ptr
                self.features[slot] = key
                self.labels[slot] = class_id
                self.class_ptrs[class_id] = (local_ptr + 1) % class_capacity
                self.class_counts[class_id] = min(
                    class_capacity, self.class_counts[class_id] + 1
                )

        self.is_ready = sum(self.class_counts) == self.capacity

    def state_dict(self):
        """Return a device-independent snapshot for exact interrupted-stage resume."""
        return {
            'capacity': self.capacity,
            'num_classes': self.num_classes,
            'class_capacities': list(self.class_capacities),
            'class_ptrs': list(self.class_ptrs),
            'class_counts': list(self.class_counts),
            'is_ready': bool(self.is_ready),
            'features': None if self.features is None else self.features.detach().cpu(),
            'labels': None if self.labels is None else self.labels.detach().cpu(),
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        """Restore a queue snapshot while rejecting incompatible configurations."""
        if int(state['capacity']) != self.capacity or int(state['num_classes']) != self.num_classes:
            raise ValueError(
                "The saved teacher-feature queue is incompatible with the current "
                f"configuration: saved capacity/classes={state['capacity']}/"
                f"{state['num_classes']}, current={self.capacity}/{self.num_classes}."
            )
        if list(state['class_capacities']) != self.class_capacities:
            raise ValueError("The saved per-class queue quotas do not match the current quotas.")

        saved_features = state.get('features')
        saved_labels = state.get('labels')
        self.features = None if saved_features is None else saved_features.to(self.device)
        self.labels = None if saved_labels is None else saved_labels.to(self.device)
        self.class_ptrs = [int(value) for value in state['class_ptrs']]
        self.class_counts = [int(value) for value in state['class_counts']]
        self.is_ready = bool(state.get('is_ready', sum(self.class_counts) == self.capacity))

class DecoupledLACDLoss(nn.Module):
    def __init__(self, temp_pos=0.1, temp_neg=0.07, use_hard_mining=USE_HARD_MINING, top_k=LACD_TOP_K,
                 use_focal_weighting=USE_FOCAL_WEIGHTING, focal_gamma=FOCAL_GAMMA, focal_margin=FOCAL_MARGIN):
        super(DecoupledLACDLoss, self).__init__()
        self.temp_pos = temp_pos
        self.temp_neg = temp_neg
        self.use_hard_mining = use_hard_mining
        self.top_k = top_k
        self.use_focal_weighting = use_focal_weighting
        self.focal_gamma = focal_gamma
        self.focal_margin = focal_margin

        self.epoch_distances = []

    # 【修正】增加 queue_features=None, queue_labels=None
    def forward(self, z_adv_s, z_clean_t, labels, queue_features=None,
                queue_labels=None, instance_weights=None):
        z_adv_s_norm = F.normalize(z_adv_s, dim=1, p=2, eps=1e-8)
        z_clean_t_norm = F.normalize(z_clean_t, dim=1, p=2, eps=1e-8)

        distances = torch.norm(z_adv_s_norm - z_clean_t_norm, p=2, dim=1)
        with torch.no_grad():
            self.epoch_distances.extend(distances.detach().cpu().numpy().tolist())

        sim_matrix = torch.matmul(z_adv_s_norm, z_clean_t_norm.t())

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(z_adv_s.device)
        neg_mask_batch = (mask == 0).float()

        sim_pos = torch.diag(sim_matrix).view(-1, 1)

        if USE_SOFTPLUS_MARGIN:
            sim_pos_eff = SOFTPLUS_SIM_TARGET - F.softplus(
                SOFTPLUS_BETA * (SOFTPLUS_SIM_TARGET - sim_pos)) / SOFTPLUS_BETA
        else:
            sim_pos_eff = sim_pos

        # 正样本除以固定温度 (应用了防坍缩边界的有效相似度)
        logits_pos = sim_pos_eff / self.temp_pos

        # =======================================================
        # 【修正】为固定温度的 Decoupled Loss 也加上队列支持
        # =======================================================
        if queue_features is not None and queue_labels is not None:
            valid_mask = queue_labels != -1
            if valid_mask.any():
                valid_q_feats = queue_features[valid_mask]
                valid_q_labels = queue_labels[valid_mask].view(1, -1)

                sim_queue = torch.matmul(z_adv_s_norm, valid_q_feats.t())
                mask_queue = torch.eq(labels, valid_q_labels).float()
                neg_mask_queue = (mask_queue == 0).float()

                # [新增] 全局拼接 Batch 和 Queue
                sim_all = torch.cat([sim_matrix, sim_queue], dim=1)
                neg_mask_all = torch.cat([neg_mask_batch, neg_mask_queue], dim=1)
            else:
                sim_all = sim_matrix
                neg_mask_all = neg_mask_batch
        else:
            sim_all = sim_matrix
            neg_mask_all = neg_mask_batch
            # =======================================================

        # -----------------------------------------------------------------
        # [核心修复] 数值绝对安全的 LogSumExp 逻辑
        # -----------------------------------------------------------------
        # 1. 计算负样本 Logits
        logits_neg_all = sim_all / self.temp_neg

        if self.use_focal_weighting:
            weight_neg = F.relu(sim_all.detach() - self.focal_margin) ** self.focal_gamma
            logits_neg_all = logits_neg_all + torch.log(weight_neg + 1e-8)

        # 2. 屏蔽正样本：将其 logits 置为 -1e9，防止被 Top-K 选中，且 e^(-1e9) = 0
        logits_neg_all = logits_neg_all.masked_fill(neg_mask_all == 0, -1e9)

        # 3. 全局 Top-K 挖掘 (直接在 Logits 层面选，不涉及 exp，绝对安全)
        if self.use_hard_mining:
            real_k = min(self.top_k, logits_neg_all.size(1))
            topk_logits_neg, _ = torch.topk(logits_neg_all, k=real_k, dim=1)
        else:
            topk_logits_neg = logits_neg_all

        # 4. 拼接 正样本 logits (索引0) 和 负样本 logits，复用内置 CE 享受完美防溢出
        all_logits = torch.cat([logits_pos, topk_logits_neg], dim=1)
        targets = torch.zeros(logits_pos.size(0), dtype=torch.long, device=z_adv_s.device)

        # [修改]：不直接求均值，而是保持实例维度
        loss_ce_unreduced = F.cross_entropy(all_logits, targets, reduction='none')

        # [新增]：应用实例级动态置信度权重
        if instance_weights is not None:
            # instance_weights 的 shape 应该是 [Batch]
            loss_ce = (loss_ce_unreduced * instance_weights).mean()
        else:
            loss_ce = loss_ce_unreduced.mean()

        # =================================================================
        # [进阶方案] 方差正则化 (强制要求维度丰富度)
        # =================================================================
        if USE_VARIANCE_REG:
            # 强迫 Student 特征在 Batch 内维持高方差，防止缩成一点
            std_z_s = torch.sqrt(z_adv_s_norm.var(dim=0) + 1e-4)
            loss_var = torch.mean(F.relu(VAR_GAMMA - std_z_s))
            return loss_ce + VAR_WEIGHT * loss_var

        return loss_ce

class AdaptiveDecoupledLACDLoss(nn.Module):
    """
    [Original Idea 2] 实例级自适应弹性对比损失
    Vulnerability-Aware Elastic Distillation Loss
    """

    def __init__(self, temp_pos_min=TEMP_POS_MIN, temp_pos_max=TEMP_POS_MAX, temp_neg=LACD_TEMP_NEG,
                 use_hard_mining=USE_HARD_MINING, top_k=LACD_TOP_K,
                 use_focal_weighting=USE_FOCAL_WEIGHTING, focal_gamma=FOCAL_GAMMA, focal_margin=FOCAL_MARGIN):
        super(AdaptiveDecoupledLACDLoss, self).__init__()
        self.temp_pos_min = temp_pos_min
        self.temp_pos_max = temp_pos_max
        self.temp_neg = temp_neg
        self.use_hard_mining = use_hard_mining
        self.top_k = top_k
        self.use_focal_weighting = use_focal_weighting
        self.focal_gamma = focal_gamma
        self.focal_margin = focal_margin

        self.batch_t_min = 0.0
        self.batch_t_max = 0.0
        self.batch_t_mean = 0.0

        # [新增代码] 距离分布跟踪器
        self.epoch_distances = []

        # =======================================================
        # [新增] EMA 动态边界缓冲区 (buffer 不会随着优化器更新)
        # =======================================================
        self.register_buffer('running_d_min', torch.tensor(0.0))
        self.register_buffer('running_d_max', torch.tensor(1.0))
        self.ema_momentum = 0.9  # 平滑系数：0.9 表示依赖 90% 的历史 + 10% 的当下
        self.is_first_batch = True

    def forward(self, z_adv_s, z_clean_t, labels, queue_features=None,
                queue_labels=None, instance_weights=None, temperature_groups=None):
        # 1. 特征归一化
        z_adv_s_norm = F.normalize(z_adv_s, dim=1, p=2, eps=1e-8)
        z_clean_t_norm = F.normalize(z_clean_t, dim=1, p=2, eps=1e-8)

        # 2. 计算实例级距离 (L2 距离) 评估样本脆弱性
        # 方案一，Batch Min-Max
        distances = torch.norm(z_adv_s_norm - z_clean_t_norm, p=2, dim=1)  # Shape: [Batch]
        # [新增代码] 收集当前 batch 的距离用于后续分析 (detach 防止内存泄漏，转到 CPU 不影响显存)
        with torch.no_grad():
            self.epoch_distances.extend(distances.detach().cpu().numpy().tolist())

        # Min-Max 归一化到 [0, 1] (加上 eps 防止除零)
        if temperature_groups is None:
            temperature_groups = (len(distances),)
        temperature_groups = tuple(int(size) for size in temperature_groups)
        if any(size <= 0 for size in temperature_groups) or sum(temperature_groups) != len(distances):
            raise ValueError(
                "temperature_groups must be positive and sum to the query batch size"
            )
        # Current and replay views retain one joint contrastive negative pool,
        # while each view group calibrates its own instance temperatures. This
        # removes mixed-source min/max coupling without changing the negatives.
        normalized_groups = []
        offset = 0
        for size in temperature_groups:
            group = distances[offset:offset + size]
            d_min, d_max = group.min(), group.max()
            normalized_groups.append(torch.clamp(
                (group - d_min) / (d_max - d_min + 1e-4),
                min=0.0, max=1.0,
            ))
            offset += size
        d_norm = torch.cat(normalized_groups)

        # 方案二，全局固定阈值
        # distances = torch.norm(z_adv_s_norm - z_clean_t_norm, p=2, dim=1)  # Shape: [Batch]
        #
        # # [新增代码] 收集当前 batch 的距离用于后续分析 (detach 防止内存泄漏，转到 CPU 不影响显存)
        # with torch.no_grad():
        #     self.epoch_distances.extend(distances.detach().cpu().numpy().tolist())
        #
        # # [暂时修改] 建议先将这个硬编码的 1.0 改为一个保守值，比如 1.414 (这是正交特征的距离)
        # # 等我们画出图之后，再把它改成最科学的数值
        # d_norm = torch.clamp(distances / 1.4, min=0.0, max=1.0)

        # # 方案三 EMA 动态全局 Min-Max
        # distances = torch.norm(z_adv_s_norm - z_clean_t_norm, p=2, dim=1)
        #
        # with torch.no_grad():
        #     self.epoch_distances.extend(distances.detach().cpu().numpy().tolist())
        #
        #     # [核心修改：使用稳健的百分位数代替绝对 Max/Min]
        #     # 考虑到 Batch Size 可能只有 64，95% 分位数 (约第 60 个样本) 比 99% 更稳定
        #     batch_min = distances.min()  # 下限用 min 一般没问题，因为不会出现负数异常值
        #     batch_p95 = torch.quantile(distances, 0.95)
        #
        #     # 维护全局动态边界 (用 95th percentile 更新上限)
        #     if self.training:
        #         if self.is_first_batch:
        #             self.running_d_min.copy_(batch_min)
        #             self.running_d_max.copy_(batch_p95)
        #             self.is_first_batch = False
        #         else:
        #             self.running_d_min.copy_(
        #                 self.ema_momentum * self.running_d_min + (1 - self.ema_momentum) * batch_min)
        #             self.running_d_max.copy_(
        #                 self.ema_momentum * self.running_d_max + (1 - self.ema_momentum) * batch_p95)
        #
        # # 使用【平滑后的 95% 分位数边界】进行归一化
        # # 此时，距离超过 95% 分位数的那 5% 极端样本，算出来的 d_norm 会大于 1
        # # 但因为外面有 Clamp(max=1.0)，它们会被完美截断，全部享受最高温度！
        # d_norm = torch.clamp((distances - self.running_d_min) / (self.running_d_max - self.running_d_min + 1e-4),
        #                      min=0.0, max=1.0)

        # 3. 动态计算每个样本的正向温度 (距离越大 -> 扭曲越严重 -> 需要的弹性温度越大)
        t_pos_dynamic = self.temp_pos_min + d_norm * (self.temp_pos_max - self.temp_pos_min)
        t_pos_dynamic = t_pos_dynamic.view(-1, 1).detach()  # Shape: [Batch, 1], 停止梯度阻断

        # # 随机温度验证
        # # 获取当前 Batch Size
        # batch_size = z_adv_s.size(0)
        # # 直接在 GPU 上生成形状为 [Batch, 1] 的 Tensor，每个样本的温度都不一样
        # t_pos_dynamic = torch.empty(batch_size, 1, device=z_adv_s.device).uniform_(self.temp_pos_min,
        #                                                                            self.temp_pos_max).detach()

        self.batch_t_min = t_pos_dynamic.min().item()
        self.batch_t_max = t_pos_dynamic.max().item()
        self.batch_t_mean = t_pos_dynamic.mean().item()

        # 4. 计算相似度矩阵与 Mask
        sim_matrix = torch.matmul(z_adv_s_norm, z_clean_t_norm.t())
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(z_adv_s.device)
        neg_mask_batch = (mask == 0).float()

        # 5. 解耦计算 Logits
        sim_pos = torch.diag(sim_matrix).view(-1, 1)

        # =================================================================
        # [核心创新] 由 Softplus 驱动的平滑相似度截断防坍缩
        # =================================================================
        if USE_SOFTPLUS_MARGIN:
            sim_pos_eff = SOFTPLUS_SIM_TARGET - F.softplus(
                SOFTPLUS_BETA * (SOFTPLUS_SIM_TARGET - sim_pos)) / SOFTPLUS_BETA
        else:
            sim_pos_eff = sim_pos

        # [核心创新] 正样本除以动态温度
        logits_pos = sim_pos_eff / t_pos_dynamic

        # =======================================================
        # 【核心修改 2】：计算与特征队列的负样本相似度
        # =======================================================
        if queue_features is not None and queue_labels is not None:
            valid_mask = queue_labels != -1
            if valid_mask.any():
                valid_q_feats = queue_features[valid_mask]
                valid_q_labels = queue_labels[valid_mask].view(1, -1)

                # 计算相似度与 Mask
                sim_queue = torch.matmul(z_adv_s_norm, valid_q_feats.t())
                mask_queue = torch.eq(labels, valid_q_labels).float()
                neg_mask_queue = (mask_queue == 0).float()

                # [新增] 全局拼接 Batch 和 Queue
                sim_all = torch.cat([sim_matrix, sim_queue], dim=1)
                neg_mask_all = torch.cat([neg_mask_batch, neg_mask_queue], dim=1)
            else:
                sim_all = sim_matrix
                neg_mask_all = neg_mask_batch
        else:
            sim_all = sim_matrix
            neg_mask_all = neg_mask_batch
            # =======================================================

        # -----------------------------------------------------------------
        # [核心修复] 数值绝对安全的 LogSumExp 逻辑
        # -----------------------------------------------------------------
        logits_neg_all = sim_all / self.temp_neg

        if self.use_focal_weighting:
            weight_neg = F.relu(sim_all.detach() - self.focal_margin) ** self.focal_gamma
            logits_neg_all = logits_neg_all + torch.log(weight_neg + 1e-8)

        logits_neg_all = logits_neg_all.masked_fill(neg_mask_all == 0, -1e9)

        if self.use_hard_mining:
            real_k = min(self.top_k, logits_neg_all.size(1))
            topk_logits_neg, _ = torch.topk(logits_neg_all, k=real_k, dim=1)
        else:
            topk_logits_neg = logits_neg_all

        all_logits = torch.cat([logits_pos, topk_logits_neg], dim=1)
        targets = torch.zeros(logits_pos.size(0), dtype=torch.long, device=z_adv_s.device)

        # [修改]：不直接求均值，而是保持实例维度
        loss_ce_unreduced = F.cross_entropy(all_logits, targets, reduction='none')

        # [新增]：应用实例级动态置信度权重
        if instance_weights is not None:
            # instance_weights 的 shape 应该是 [Batch]
            loss_ce = (loss_ce_unreduced * instance_weights).mean()
        else:
            loss_ce = loss_ce_unreduced.mean()

        # =================================================================
        # [进阶方案] 方差正则化 (彻底断绝维度坍缩)
        # =================================================================
        if USE_VARIANCE_REG:
            # 强迫 Student 特征在 Batch 内维持高方差，防止缩成一点
            std_z_s = torch.sqrt(z_adv_s_norm.var(dim=0) + 1e-4)
            loss_var = torch.mean(F.relu(VAR_GAMMA - std_z_s))
            return loss_ce + VAR_WEIGHT * loss_var

        return loss_ce

def compute_grad_cosine_sim(loss1, loss2, model):
    # [修正] 必须计算所有参数的梯度，才能与老代码的 Sim 值对齐
    params = [p for p in model.parameters() if p.requires_grad]

    if not params: return 0.0

    grads1 = torch.autograd.grad(loss1, params, retain_graph=True, allow_unused=True)
    grads2 = torch.autograd.grad(loss2, params, retain_graph=True, allow_unused=True)

    g1_list, g2_list = [], []
    for p, g1, g2 in zip(params, grads1, grads2):
        g1_list.append(g1.view(-1) if g1 is not None else torch.zeros_like(p).view(-1))
        g2_list.append(g2.view(-1) if g2 is not None else torch.zeros_like(p).view(-1))

    g1_vec = torch.cat(g1_list)
    g2_vec = torch.cat(g2_list)

    if g1_vec.norm() == 0 or g2_vec.norm() == 0: return 0.0
    return F.cosine_similarity(g1_vec.unsqueeze(0), g2_vec.unsqueeze(0), eps=1e-8).item()
