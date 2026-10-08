import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_train_mode(modules, freeze_bn: bool) -> None:
    for module in modules:
        module.train()
        if freeze_bn:
            for m in module.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()


class Averager:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def add(self, value: float) -> None:
        self.total += float(value)
        self.count += 1

    def item(self) -> float:
        return self.total / self.count if self.count else 0.0
