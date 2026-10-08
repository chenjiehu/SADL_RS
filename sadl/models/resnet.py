from pathlib import Path

import torch
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None, dilation: int = 1):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class DilatedResNet(nn.Module):
    def __init__(self, layers=(3, 4, 6), dilate=(False, True)):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2, dilate=dilate[0])
        self.layer3 = self._make_layer(256, layers[2], stride=2, dilate=dilate[1])
        self.out_channels = (256, 512, 1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, previous_dilation)]
        self.inplanes = planes * Bottleneck.expansion
        layers += [Bottleneck(self.inplanes, planes, dilation=self.dilation) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return f1, f2, f3


def resnet50_backbone(pretrained_path=None) -> DilatedResNet:
    model = DilatedResNet()
    if pretrained_path is None:
        return model
    pretrained_path = Path(pretrained_path)
    if not pretrained_path.is_file():
        raise FileNotFoundError(f'pretrained backbone not found: {pretrained_path}')
    state = torch.load(pretrained_path, map_location='cpu', weights_only=True)
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    missing = [k for k in own if k not in matched and not k.endswith('num_batches_tracked')]
    model.load_state_dict(matched, strict=False)
    print(f'backbone: loaded {len(matched)} tensors from {pretrained_path.name}, {len(missing)} left at random init')
    return model
