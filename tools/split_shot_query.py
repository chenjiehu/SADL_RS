import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sadl.config import PROJECT_ROOT


def collect_label_files(root: Path):
    folder = root / 'labels'
    if not folder.is_dir():
        raise FileNotFoundError(f'labels folder not found: {folder}')
    files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.tif'):
        files.extend(folder.glob(ext))
    if not files:
        raise ValueError(f'no label images under {folder}')
    return sorted(files)


def class_proportions(label_paths, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for path in label_paths:
        label = np.asarray(Image.open(path), dtype=np.int64)
        if label.ndim == 3:
            label = label[..., 0]
        valid = label[label < num_classes]
        counts += np.bincount(valid.ravel(), minlength=num_classes)[:num_classes]
    total = counts.sum()
    return counts / total if total else np.zeros(num_classes)


def write_names(path: Path, names):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for name in names:
            writer.writerow([name])


def split_shot_query(root: Path, shot: int, num_classes: int, num_splits: int, min_proportion: float,
                     max_attempts: int, seed: int):
    label_files = collect_label_files(root)
    shot_dir = root / f'shot_{root.name}'
    query_dir = root / f'query_{root.name}'
    shot_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator().manual_seed(seed)
    produced = 0
    for attempt in range(1, max_attempts + 1):
        if produced >= num_splits:
            break
        perm = torch.randperm(len(label_files), generator=generator).tolist()
        shot_files = [label_files[i] for i in perm[:shot]]
        query_files = [label_files[i] for i in perm[shot:]]

        proportions = class_proportions(shot_files, num_classes)
        covered = int(np.count_nonzero(proportions > 0))
        print(f'attempt {attempt}: {covered}/{num_classes} classes, min proportion {proportions.min():.4f}')
        if covered < num_classes or proportions.min() < min_proportion:
            continue

        write_names(shot_dir / f'{shot}_shot{produced}.csv', (p.stem for p in shot_files))
        write_names(query_dir / f'{shot}_query{produced}.csv', (p.stem for p in query_files))
        produced += 1

    print(f'generated {produced}/{num_splits} splits under {shot_dir} and {query_dir}')


def main():
    parser = argparse.ArgumentParser(description='generate shot/query name lists')
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT / 'dataset' / 'GID5_new')
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--num-splits', type=int, default=100)
    parser.add_argument('--min-proportion', type=float, default=0.06)
    parser.add_argument('--max-attempts', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    split_shot_query(args.root, args.shot, args.num_classes, args.num_splits, args.min_proportion,
                     args.max_attempts, args.seed)


if __name__ == '__main__':
    main()
