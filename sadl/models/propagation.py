import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_with_ignore(labels: torch.Tensor, num_classes: int, ignore_label: int) -> torch.Tensor:
    valid = (labels != ignore_label) & (labels < num_classes)
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    return F.one_hot(safe.long(), num_classes).float() * valid.unsqueeze(1)


def squared_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_sq = (a * a).sum(dim=1, keepdim=True)
    b_sq = (b * b).sum(dim=1).unsqueeze(0)
    return (a_sq + b_sq - 2 * a @ b.t()).clamp_(min=0)


def class_prototypes(features: torch.Tensor, labels: torch.Tensor, num_classes: int, ignore_label: int) -> torch.Tensor:
    valid = (labels != ignore_label) & (labels < num_classes)
    feats, labs = features[valid], labels[valid].long()
    prototypes = features.new_zeros(num_classes, features.shape[1]).index_add_(0, labs, feats)
    counts = torch.bincount(labs, minlength=num_classes).clamp_(min=1)
    return prototypes / counts.unsqueeze(1)


class SigmaNetwork(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class LabelPropagation(nn.Module):
    def __init__(self, feature_dim: int, alpha: float = 0.99, learn_alpha: bool = False,
                 edge_threshold: float = 0.9, confidence_threshold: float = 1.0, ignore_label: int = 255):
        super().__init__()
        self.sigma_net = SigmaNetwork(feature_dim)
        alpha_tensor = torch.tensor([alpha], dtype=torch.float32)
        if learn_alpha:
            self.alpha = nn.Parameter(alpha_tensor)
        else:
            self.register_buffer('alpha', alpha_tensor)
        self.edge_threshold = edge_threshold
        self.confidence_threshold = confidence_threshold
        self.ignore_label = ignore_label
        self.eps = float(np.finfo(float).eps)

    def forward(self, features: torch.Tensor, shot_labels: torch.Tensor, num_classes: int):
        n, d = features.shape
        n_shot = shot_labels.shape[0]

        sigma = self.sigma_net(features)
        emb = features / (sigma + self.eps)
        weights = torch.exp(-squared_distances(emb, emb) / d)
        weights = torch.where(weights > self.edge_threshold, weights, torch.zeros_like(weights))

        degree_inv_sqrt = torch.rsqrt(weights.sum(dim=0) + self.eps)
        propagation = degree_inv_sqrt.unsqueeze(1) * weights * degree_inv_sqrt.unsqueeze(0)

        labels_shot = one_hot_with_ignore(shot_labels, num_classes, self.ignore_label)
        labels_query = features.new_zeros(n - n_shot, num_classes)
        prior = torch.cat([labels_shot, labels_query], dim=0)

        system = torch.eye(n, device=features.device, dtype=features.dtype) - self.alpha * propagation + self.eps
        scores = torch.linalg.solve(system, prior)
        query_scores = scores[n_shot:]

        confident = query_scores.norm(dim=1) >= self.confidence_threshold
        return query_scores, query_scores.argmax(dim=1), query_scores.softmax(dim=1), confident


class SuperpixelClassifier(nn.Module):
    def __init__(self, propagation: LabelPropagation, ignore_label: int = 255):
        super().__init__()
        self.propagation = propagation
        self.ignore_label = ignore_label

    def forward(self, batch, num_classes: int):
        _, pred_label, prob, confident = self.propagation(batch.features, batch.shot_labels, num_classes)

        if confident.all():
            superpixel_out = prob
        else:
            labels = torch.cat([batch.shot_labels, pred_label], dim=0)
            keep = torch.cat([torch.ones(batch.num_shot, dtype=torch.bool, device=confident.device), confident])
            prototypes = class_prototypes(batch.features[keep], labels[keep], num_classes, self.ignore_label)
            query_features = batch.features[batch.num_shot:]
            logits = -squared_distances(query_features[~confident], prototypes)
            superpixel_out = prob.clone()
            superpixel_out[~confident] = logits

        pixel_out = superpixel_out[batch.query_index].permute(0, 3, 1, 2)
        return pixel_out, superpixel_out
