from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet50_backbone


class ASPPHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations: Sequence[int]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d) for d in dilations
        ])
        for conv in self.branches:
            nn.init.normal_(conv.weight, 0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branches[0](x)
        for branch in self.branches[1:]:
            out = out + branch(x)
        return out


class FeatureEncoder(nn.Module):
    def __init__(self, pretrained_path, out_channels: int, dilations: Sequence[int]):
        super().__init__()
        self.backbone = resnet50_backbone(pretrained_path)
        self.head = ASPPHead(sum(self.backbone.out_channels), out_channels, dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        f1, f2, f3 = self.backbone(x)
        size = f1.shape[-2:]
        f2 = F.interpolate(f2, size=size, mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, size=size, mode='bilinear', align_corners=True)
        out = self.head(torch.cat([f3, f2, f1], dim=1))
        return F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
