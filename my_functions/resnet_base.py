"""ResNet-18 with the same normalization/feature contract as UnifiedWideResNet."""
import torch
import torch.nn as nn


class UnifiedResNet18(nn.Module):
    def __init__(self, num_classes=10, dataset_mean=(0, 0, 0),
                 dataset_std=(1, 1, 1), small_input=False):
        super().__init__()
        # Lazy import leaves legacy WRN construction/RNG behavior unchanged.
        from torchvision.models import resnet18
        network = resnet18(weights=None, num_classes=num_classes)
        self.register_buffer("mean", torch.tensor(dataset_mean).float().view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(dataset_std).float().view(1, 3, 1, 1))
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool
        if small_input:
            # CIFAR: keep spatial resolution at the stem. Residual stages keep
            # the standard [2, 2, 2, 2] BasicBlock configuration.
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
            self.maxpool = nn.Identity()
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.layer4 = network.layer4
        self.avgpool = network.avgpool
        # Shared callers use the classifier name ``linear``.
        self.linear = network.fc
        self.use_mlp_projector = False

    def forward(self, x, return_feature=False):
        # Attacks continue to operate on [0, 1] images; normalize exactly once.
        x = (x - self.mean) / self.std
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = torch.flatten(self.avgpool(x), 1)
        logits = self.linear(features)
        if return_feature:
            # PACE LACD uses item 2; final feature plots use item 1.
            return logits, features, features, None
        return logits
