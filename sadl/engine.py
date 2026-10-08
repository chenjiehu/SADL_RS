from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data.datasets import SegmentationDataset, load_all
from .data.superpixel import refined_slic, slic_segments
from .metrics import confusion_matrix, summarize
from .models import build_models


MODEL_NAMES = ('extractor', 'adapter', 'classifier')


def forward_episode(models, shot_images, shot_labels, query_images, query_labels, segments, num_classes,
                    use_adapter=True):
    extractor, adapter, classifier = models
    batch, _ = extractor(shot_images, shot_labels, query_images, query_labels, segments)
    if use_adapter:
        adapted, _ = adapter(batch.features)
        batch = batch.with_features(adapted)
    pred, _ = classifier(batch, num_classes)
    return pred, batch


def save_checkpoint(path, models, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: m.state_dict() for name, m in zip(MODEL_NAMES, models)}
    state.update(extra)
    torch.save(state, path)
    print(f'checkpoint saved: {path}')


def load_checkpoint(path, models, device, strict=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'checkpoint not found: {path}')
    state = torch.load(path, map_location=device, weights_only=True)
    for name, m in zip(MODEL_NAMES, models):
        m.load_state_dict(state[name], strict=strict)
    print(f'checkpoint loaded: {path}')
    return {k: v for k, v in state.items() if k not in MODEL_NAMES}


def build_and_init(cfg, device):
    models = build_models(cfg, device)
    if cfg.init_from:
        load_checkpoint(cfg.init_from, models, device)
    return models


def load_shot_set(cfg, split, transform, device):
    shot_list = cfg.target_root / cfg.shot_dir / f'{cfg.shot}_shot{split}.csv'
    dataset = SegmentationDataset(cfg.target_root, shot_list, cfg.target_img_ext, cfg.target_label_ext,
                                  transform, cfg.crop_size)
    images, labels, names = load_all(dataset)
    return images.to(device), labels.to(device).long(), names


def build_query_set(cfg, split, transform):
    query_list = cfg.target_root / cfg.query_dir / f'{cfg.shot}_query{split}.csv'
    return SegmentationDataset(cfg.target_root, query_list, cfg.target_img_ext, cfg.target_label_ext,
                               transform, cfg.crop_size)


def save_predictions(pred, names, pred_dir):
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    masks = pred.argmax(1).to(torch.uint8).cpu().numpy()
    for mask, name in zip(masks, names):
        Image.fromarray(mask, mode='L').save(pred_dir / f'{name}.png')


@torch.no_grad()
def evaluate(cfg, device, models, shot_images, shot_labels, query_loader, max_batches=None, pred_dir=None):
    for m in models:
        m.eval()
    if torch.unique(shot_labels).numel() < cfg.target_classes:
        print('shot set does not cover all target classes, skip evaluation')
        return None

    segments_shot = refined_slic(shot_images, shot_labels, cfg.n_segments, cfg.slic_compactness)
    confusion = torch.zeros(cfg.target_classes, cfg.target_classes, dtype=torch.long, device=device)

    for it, (images_q, labels_q, names) in enumerate(query_loader):
        if max_batches is not None and it >= max_batches:
            break
        images_q = images_q.to(device, non_blocking=True)
        labels_q = labels_q.to(device, non_blocking=True).long()
        placeholder = torch.full_like(labels_q, cfg.ignore_label)
        segments = torch.cat([segments_shot, slic_segments(images_q, cfg.n_segments, cfg.slic_compactness)], 0)
        pred, _ = forward_episode(models, shot_images, shot_labels, images_q, placeholder, segments,
                                  cfg.target_classes, cfg.use_adapter_in_eval)
        confusion += confusion_matrix(pred.argmax(1), labels_q, cfg.target_classes, cfg.ignore_label)
        if pred_dir is not None:
            save_predictions(pred, names, pred_dir)

    return summarize(confusion.cpu().numpy())


def save_metrics(result_dir, prefix, metrics):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(result_dir / f'{prefix}_confusion.csv', metrics.confusion, fmt='%d', delimiter=',')
    np.savetxt(result_dir / f'{prefix}_iou.csv', metrics.iou[None], fmt='%.6f', delimiter=',')
    np.savetxt(result_dir / f'{prefix}_f1.csv', metrics.f1[None], fmt='%.6f', delimiter=',')
    with (result_dir / f'{prefix}_summary.txt').open('w', encoding='utf-8') as f:
        f.write(str(metrics) + '\n')


def print_summary(results, title):
    if not results:
        return
    print(f'\n==== {title} ====')
    for split, m in results.items():
        print(f'split {split}: mIoU {m.miou:.4f} | mF1 {m.mf1:.4f} | OA {m.oa:.4f}')
    print(f'mean: mIoU {np.mean([m.miou for m in results.values()]):.4f} | '
          f'mF1 {np.mean([m.mf1 for m in results.values()]):.4f} | '
          f'OA {np.mean([m.oa for m in results.values()]):.4f}')
