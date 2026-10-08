from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def remap_labels(labels: torch.Tensor, mapping: Sequence[int], table_size: int = 256) -> torch.Tensor:
    lut = torch.arange(table_size, device=labels.device, dtype=labels.dtype)
    for src, dst in enumerate(mapping):
        lut[src] = dst
    return lut[labels.long()]


@torch.no_grad()
def random_convolution(images: torch.Tensor, prob: float, kernels: Sequence[int]) -> torch.Tensor:
    if np.random.rand() <= prob:
        return images
    k = int(kernels[np.random.randint(len(kernels))])
    weight = torch.empty(3, 3, k, k, device=images.device, dtype=images.dtype)
    nn.init.xavier_normal_(weight)
    return F.conv2d(images, weight, padding=k // 2)


def paste_target_regions(images_s: torch.Tensor, labels_s: torch.Tensor,
                         images_t: torch.Tensor, labels_t: torch.Tensor,
                         values: Sequence[int]):
    if images_s.shape[0] != images_t.shape[0]:
        raise ValueError(f'batch mismatch: source {images_s.shape[0]} vs target {images_t.shape[0]}')
    mask = torch.isin(labels_t, torch.as_tensor(values, device=labels_t.device))
    images = torch.where(mask.unsqueeze(1), images_t, images_s)
    labels = torch.where(mask, labels_t, labels_s)
    return images, labels


def gradient_ascent(shot_images, shot_labels, query_images, query_labels, segments,
                    extractor, classifier, num_classes: int, criterion,
                    steps: int = 3, lr: float = 1e-4):
    shot_images = shot_images.detach().clone().requires_grad_(True)
    query_images = query_images.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([shot_images, query_images], lr=lr)

    modes = [(m, m.training) for net in (extractor, classifier) for m in net.modules()]
    extractor.eval()
    classifier.eval()
    for _ in range(steps):
        optimizer.zero_grad()
        batch, _ = extractor(shot_images, shot_labels, query_images, query_labels, segments)
        pred, _ = classifier(batch, num_classes)
        loss = criterion(pred, query_labels)
        (-loss).backward()
        optimizer.step()
    for module, training in modes:
        module.training = training
    return shot_images.detach(), query_images.detach()
