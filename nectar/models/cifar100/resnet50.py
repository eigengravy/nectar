import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1_weight = nn.Parameter(torch.ones(out_channels))
        self.bn1_bias = nn.Parameter(torch.zeros(out_channels))
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2_weight = nn.Parameter(torch.ones(out_channels))
        self.bn2_bias = nn.Parameter(torch.zeros(out_channels))
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3_weight = nn.Parameter(torch.ones(out_channels * self.expansion))
        self.bn3_bias = nn.Parameter(torch.zeros(out_channels * self.expansion))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = F.batch_norm(
            out,
            running_mean=None,
            running_var=None,
            weight=self.bn1_weight,
            bias=self.bn1_bias,
            training=True,
        )
        out = self.relu(out)

        out = self.conv2(out)
        out = F.batch_norm(
            out,
            running_mean=None,
            running_var=None,
            weight=self.bn2_weight,
            bias=self.bn2_bias,
            training=True,
        )
        out = self.relu(out)

        out = self.conv3(out)
        out = F.batch_norm(
            out,
            running_mean=None,
            running_var=None,
            weight=self.bn3_weight,
            bias=self.bn3_bias,
            training=True,
        )

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=100):  # CIFAR-100 has 100 classes
        super(ResNet50, self).__init__()
        self.in_channels = 64

        # Adjusted for CIFAR-100's 32x32 input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_weight = nn.Parameter(torch.ones(64))
        self.bn1_bias = nn.Parameter(torch.zeros(64))
        self.relu = nn.ReLU(inplace=True)
        # Removed maxpool to preserve spatial dimensions

        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * Bottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.batch_norm(
            x,
            running_mean=None,
            running_var=None,
            weight=self.bn1_weight,
            bias=self.bn1_bias,
            training=True,
        )
        x = self.relu(x)
        # Removed maxpool

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
