from dataclasses import dataclass

import numpy as np
import torch


@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_label: int = 255) -> torch.Tensor:
    pred = pred.reshape(-1).long()
    target = target.reshape(-1).long()
    valid = (target != ignore_label) & (target < num_classes) & (pred < num_classes)
    index = target[valid] * num_classes + pred[valid]
    return torch.bincount(index, minlength=num_classes ** 2).view(num_classes, num_classes)


@dataclass
class SegmentationMetrics:
    iou: np.ndarray
    miou: float
    recall: np.ndarray
    precision: np.ndarray
    f1: np.ndarray
    mf1: float
    oa: float
    confusion: np.ndarray

    def __str__(self) -> str:
        return (
            f'mIoU {self.miou:.4f} | mF1 {self.mf1:.4f} | OA {self.oa:.4f}\n'
            f'IoU  {np.array2string(self.iou, precision=4)}\n'
            f'F1   {np.array2string(self.f1, precision=4)}'
        )


def summarize(confusion) -> SegmentationMetrics:
    conf = np.asarray(confusion, dtype=np.float64)
    eps = 1e-10
    tp = np.diag(conf)
    gt = conf.sum(axis=1)
    pr = conf.sum(axis=0)
    iou = tp / (gt + pr - tp + eps)
    recall = tp / (gt + eps)
    precision = tp / (pr + eps)
    f1 = 2 * recall * precision / (recall + precision + eps)
    return SegmentationMetrics(
        iou=iou,
        miou=float(iou.mean()),
        recall=recall,
        precision=precision,
        f1=f1,
        mf1=float(f1.mean()),
        oa=float(tp.sum() / (conf.sum() + eps)),
        confusion=conf.astype(np.int64),
    )
