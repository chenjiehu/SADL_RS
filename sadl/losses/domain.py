from typing import Sequence

import torch


def linear_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = min(x.shape[0], y.shape[0])
    delta = x[:n] - y[:n]
    return (delta @ delta.t()).mean()


def coral(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    d = x.shape[1]
    cov_x = torch.cov(x.t())
    cov_y = torch.cov(y.t())
    return (cov_x - cov_y).pow(2).sum().sqrt() / (4 * d * d)


def domain_alignment_loss(src_feats: torch.Tensor, src_labels: torch.Tensor,
                          tgt_feats: torch.Tensor, tgt_labels: torch.Tensor,
                          common_src: Sequence[int], common_tgt: Sequence[int],
                          min_samples: int = 10, per_class_weight: float = 0.3) -> torch.Tensor:
    per_class = src_feats.new_zeros(())
    src_index, tgt_index = [], []
    for ls, lt in zip(common_src, common_tgt):
        si = (src_labels == ls).nonzero(as_tuple=True)[0]
        ti = (tgt_labels == lt).nonzero(as_tuple=True)[0]
        if si.numel() < min_samples or ti.numel() < min_samples:
            continue
        per_class = per_class + linear_mmd(src_feats[si], tgt_feats[ti]) + coral(src_feats[si], tgt_feats[ti])
        src_index.append(si)
        tgt_index.append(ti)
    if not src_index:
        return per_class
    joint = linear_mmd(src_feats[torch.cat(src_index)], tgt_feats[torch.cat(tgt_index)])
    return per_class_weight * per_class + joint
