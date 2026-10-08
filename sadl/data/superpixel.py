import numpy as np
import torch
from skimage.segmentation import slic


@torch.no_grad()
def slic_segments(images: torch.Tensor, n_segments: int, compactness: float = 20.0) -> torch.Tensor:
    segments = [
        slic(img.permute(1, 2, 0).detach().cpu().numpy(), n_segments=n_segments,
             compactness=compactness, enforce_connectivity=True, convert2lab=True)
        for img in images
    ]
    return torch.from_numpy(np.stack(segments).astype(np.int64)).to(images.device)


@torch.no_grad()
def refine_with_labels(segments: torch.Tensor, labels: torch.Tensor, label_bins: int = 256) -> torch.Tensor:
    keys = segments * label_bins + labels.long()
    return torch.stack([torch.unique(k, return_inverse=True)[1] for k in keys])


@torch.no_grad()
def refined_slic(images: torch.Tensor, labels: torch.Tensor, n_segments: int, compactness: float = 20.0) -> torch.Tensor:
    return refine_with_labels(slic_segments(images, n_segments, compactness), labels)
