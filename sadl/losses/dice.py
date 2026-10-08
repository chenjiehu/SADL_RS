import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, ignore_label: int = 255, smooth: float = 1.0):
        super().__init__()
        self.ignore_label = ignore_label
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        valid = target != self.ignore_label
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        one_hot = F.one_hot(safe_target.long(), num_classes).permute(0, 3, 1, 2).to(logits.dtype)
        one_hot = one_hot * valid.unsqueeze(1)
        prob = logits.softmax(dim=1)
        intersection = (prob * one_hot).sum(dim=(0, 2, 3))
        cardinality = prob.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1 - dice).mean()
