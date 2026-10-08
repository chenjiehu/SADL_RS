from dataclasses import dataclass, replace
from typing import List, Tuple

import torch
import torch.nn as nn

from .encoder import FeatureEncoder
from .modules import SelfAttention


@dataclass
class SuperpixelBatch:
    features: torch.Tensor
    shot_labels: torch.Tensor
    query_labels: torch.Tensor
    query_index: torch.Tensor
    num_shot: int

    def with_features(self, features: torch.Tensor) -> 'SuperpixelBatch':
        return replace(self, features=features)


def pool_superpixels(feature_map: torch.Tensor, labels: torch.Tensor, segments: torch.Tensor,
                     label_bins: int = 256) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    n, c, h, w = feature_map.shape
    flat_features = feature_map.permute(0, 2, 3, 1).reshape(n, h * w, c)
    flat_labels = labels.reshape(n, -1).long()
    flat_segments = segments.reshape(n, -1)

    pooled, majority, index_maps, counts = [], [], [], []
    offset = 0
    for i in range(n):
        _, inverse = torch.unique(flat_segments[i], return_inverse=True)
        k = int(inverse.max().item()) + 1
        summed = feature_map.new_zeros(k, c).index_add_(0, inverse, flat_features[i])
        sizes = torch.bincount(inverse, minlength=k).clamp_(min=1)
        pooled.append(summed / sizes.unsqueeze(1))
        histogram = torch.bincount(inverse * label_bins + flat_labels[i], minlength=k * label_bins)
        majority.append(histogram.view(k, label_bins).argmax(dim=1))
        index_maps.append(inverse.view(h, w) + offset)
        counts.append(k)
        offset += k
    return torch.cat(pooled), torch.cat(majority), torch.stack(index_maps), counts


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_path, feature_dim: int, aspp_dilations):
        super().__init__()
        self.encoder = FeatureEncoder(pretrained_path, feature_dim, aspp_dilations)
        self.attention = SelfAttention(feature_dim)

    def forward(self, shot_images, shot_labels, query_images, query_labels, segments):
        n_shot = shot_images.shape[0]
        images = torch.cat([shot_images, query_images], dim=0)
        labels = torch.cat([shot_labels, query_labels], dim=0)
        feature_map = self.encoder(images)
        features, sp_labels, index_map, counts = pool_superpixels(feature_map, labels, segments)
        features = self.attention(features)
        num_shot = sum(counts[:n_shot])
        batch = SuperpixelBatch(
            features=features,
            shot_labels=sp_labels[:num_shot],
            query_labels=sp_labels[num_shot:],
            query_index=index_map[n_shot:] - num_shot,
            num_shot=num_shot,
        )
        return batch, feature_map
