from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_name_list(list_path: Path) -> List[str]:
    list_path = Path(list_path)
    if not list_path.is_file():
        raise FileNotFoundError(f'name list not found: {list_path}')
    return [line.strip().split(',')[0] for line in list_path.read_text(encoding='utf-8').splitlines() if line.strip()]


class SegmentationDataset(Dataset):
    def __init__(self, root, list_path, img_ext: str, label_ext: str, transform, size: int):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f'dataset root not found: {root}')
        self.items: List[Tuple[Path, Path, str]] = [
            (root / 'images' / f'{name}{img_ext}', root / 'labels' / f'{name}{label_ext}', name)
            for name in read_name_list(list_path)
        ]
        self.transform = transform
        self.size = (size, size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        img_path, label_path, name = self.items[index]
        image = Image.open(img_path).convert('RGB').resize(self.size, Image.BICUBIC)
        label = Image.open(label_path).resize(self.size, Image.NEAREST)
        image = np.asarray(image, dtype=np.float32)
        label = np.asarray(label, dtype=np.float32)
        image, label = self.transform(image, label)
        return image, label, name


def load_all(dataset: SegmentationDataset):
    images, labels, names = zip(*(dataset[i] for i in range(len(dataset))))
    return torch.stack(images), torch.stack(labels), list(names)
