import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 统一的数据集归一化参数
# ==========================================
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2761)

TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD = (0.2302, 0.2265, 0.2262)

IMAGENETTE_MEAN = (0.4850, 0.4560, 0.4060)
IMAGENETTE_STD = (0.2290, 0.2240, 0.2250)


# ==========================================
# 2. 统一的 WideResNet 骨干网络
# ==========================================
class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate=0.3, stride=1):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout_rate = dropout_rate
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class UnifiedWideResNet(nn.Module):
    """
    统一的 WideResNet-34-10。
    1. 内置归一化层，外部传入的图像必须是 [0, 1] 范围。
    2. 支持 return_feature=True，完美兼容 LACD 的特征对齐需求。
    3. 支持 MLP Projector 的按需开启。
    """

    def __init__(self, depth=34, widen_factor=10, num_classes=10, dropout_rate=0.3,
                 dataset_mean=(0, 0, 0), dataset_std=(1, 1, 1), use_mlp_projector=False):
        super(UnifiedWideResNet, self).__init__()

        # 注册均值和方差为 buffer，保证它们会随模型在设备间移动，且不参与梯度更新
        self.register_buffer('mean', torch.tensor(dataset_mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(dataset_std).view(1, 3, 1, 1))

        self.in_planes = 16
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)

        self.use_mlp_projector = use_mlp_projector
        if self.use_mlp_projector:
            self.projector = nn.Sequential(
                nn.Linear(nStages[3], nStages[3]),
                nn.BatchNorm1d(nStages[3]),
                nn.ReLU(inplace=True),
                nn.Linear(nStages[3], 128)
            )
            self.proj_classifier = nn.Linear(128, num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        # 核心：网络内部自动处理归一化，外部攻击算法只管在 [0, 1] 内生成对抗样本
        x = (x - self.mean) / self.std

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        # Adaptive pooling is identical to 8x8 pooling for CIFAR's final 8x8
        # feature map, while also supporting Tiny-ImageNet (64px) and
        # Imagenette (160px) without changing the classifier feature width.
        out = F.adaptive_avg_pool2d(out, 1)
        feature = out.view(out.size(0), -1)
        logits = self.linear(feature)

        if return_feature:
            if self.use_mlp_projector:
                proj = self.projector(feature)
                proj_logits = self.proj_classifier(proj)
                return logits, feature, proj, proj_logits
            else:
                return logits, feature, feature, None

        return logits


# ==========================================
# 3. 统一的 EMA (Exponential Moving Average)
# ==========================================
class EMA:
    """
    模型权重平滑机制。如果在对比实验中 LACD 开启了 EMA，MART 和 TRADES 也必须开启，或者全体关闭。
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        # 仅对 requires_grad 的参数进行 EMA (排除 BN 的 running stats 等)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        if self.decay <= 0.0:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        if self.decay <= 0.0:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        if self.decay <= 0.0:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        device = next(self.model.parameters()).device
        self.shadow = {k: v.to(device) for k, v in state_dict.items()}
