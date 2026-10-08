import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Linear(channels, channels // 2)
        self.key = nn.Linear(channels, channels // 2)
        self.value = nn.Linear(channels, channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = torch.softmax(self.query(x) @ self.key(x).t(), dim=-1)
        return self.gamma * (attention @ self.value(x)) + x


class ResidualFeatureAdapter(nn.Module):
    def __init__(self, channels: int, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(channels, channels) for _ in range(depth)]
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        residual = self.net(x)
        return x + residual, residual
