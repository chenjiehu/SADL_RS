import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sadl.config import parse_config, describe

cfg = parse_config()
os.environ.setdefault('CUDA_VISIBLE_DEVICES', cfg.gpu)

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from sadl.augment import gradient_ascent, paste_target_regions, random_convolution, remap_labels
from sadl.data.datasets import SegmentationDataset
from sadl.data.superpixel import refined_slic
from sadl.data.transforms import build_transforms
from sadl.engine import (build_and_init, build_query_set, evaluate, forward_episode, load_shot_set,
                         print_summary, save_checkpoint, save_metrics)
from sadl.losses import DiceLoss, domain_alignment_loss
from sadl.metrics import confusion_matrix, summarize
from sadl.utils import Averager, count_parameters, set_seed, set_train_mode

torch.multiprocessing.set_sharing_strategy('file_system')


def train_one_epoch(cfg, device, models, optimizer, scheduler, source_loader, shot_images_t, shot_labels_t,
                    criterion, dice, epoch):
    extractor, _, classifier = models
    set_train_mode(models, cfg.freeze_bn)
    joint_classes = cfg.joint_classes
    loss_meter = Averager()
    confusion = torch.zeros(joint_classes, joint_classes, dtype=torch.long, device=device)

    for it, (images_s, labels_s, _) in enumerate(source_loader):
        if it >= cfg.iters_per_epoch:
            break
        images_s = images_s.to(device, non_blocking=True)
        labels_s = labels_s.to(device, non_blocking=True).long()

        shot_labels_remapped = remap_labels(shot_labels_t, cfg.target_label_remap)
        images_t = torch.cat([shot_images_t, shot_images_t + cfg.noise_std * torch.randn_like(shot_images_t)], 0)
        labels_t = torch.cat([shot_labels_remapped, shot_labels_remapped], 0)
        images_t = random_convolution(images_t, cfg.random_conv_prob, cfg.random_conv_kernels)
        segments_t = refined_slic(images_t, labels_t, cfg.n_segments, cfg.slic_compactness)

        shot_t, query_t = images_t[:cfg.shot], images_t[cfg.shot:]
        shot_lt, query_lt = labels_t[:cfg.shot], labels_t[cfg.shot:]
        shot_t, query_t = gradient_ascent(shot_t, shot_lt, query_t, query_lt, segments_t, extractor, classifier,
                                          joint_classes, criterion, cfg.max_phase_steps, cfg.max_phase_lr)

        images_s, labels_s = paste_target_regions(images_s, labels_s, images_t, labels_t, cfg.replace_values)
        segments_s = refined_slic(images_s, labels_s, cfg.n_segments, cfg.slic_compactness)
        shot_s, query_s = images_s[:cfg.shot], images_s[cfg.shot:]
        shot_ls, query_ls = labels_s[:cfg.shot], labels_s[cfg.shot:]

        if torch.unique(shot_ls).numel() < cfg.min_shot_classes_train:
            continue

        optimizer.zero_grad(set_to_none=True)
        pred_s, batch_s = forward_episode(models, shot_s, shot_ls, query_s, query_ls, segments_s, joint_classes)
        pred_t, batch_t = forward_episode(models, shot_t, shot_lt, query_t, query_lt, segments_t, joint_classes)

        loss_domain = domain_alignment_loss(
            batch_s.features[batch_s.num_shot:], batch_s.query_labels,
            batch_t.features[batch_t.num_shot:], batch_t.query_labels,
            cfg.common_labels_source, cfg.common_labels_target, cfg.domain_min_samples,
        )
        loss = (criterion(pred_s, query_ls) + dice(pred_s, query_ls)
                + criterion(pred_t, query_lt) + dice(pred_t, query_lt)
                + cfg.domain_loss_weight * loss_domain)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_meter.add(loss.item())
        confusion += confusion_matrix(pred_s.argmax(1), query_ls, joint_classes, cfg.ignore_label)

    metrics = summarize(confusion.cpu().numpy())
    print(f'[epoch {epoch}] train loss {loss_meter.item():.4f}')
    print(f'[epoch {epoch}] source\n{metrics}')
    return metrics


def run_split(cfg, device, models, source_loader, train_tf, test_tf, criterion, dice, split):
    shot_images_tr, shot_labels_tr, _ = load_shot_set(cfg, split, train_tf, device)
    shot_images_te, shot_labels_te, _ = load_shot_set(cfg, split, test_tf, device)
    query_loader = DataLoader(build_query_set(cfg, split, test_tf), batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    params = [p for m in models for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=list(cfg.lr_milestones), gamma=cfg.lr_gamma)

    best = None
    pred_dir = cfg.result_dir / 'predictions' / f'split{split}' if cfg.save_pred else None
    for epoch in range(cfg.epochs):
        train_one_epoch(cfg, device, models, optimizer, scheduler, source_loader, shot_images_tr, shot_labels_tr,
                        criterion, dice, epoch)
        metrics = evaluate(cfg, device, models, shot_images_te, shot_labels_te, query_loader, cfg.max_eval_batches)
        if metrics is None:
            continue
        print(f'[epoch {epoch}] target\n{metrics}')
        if best is None or metrics.miou > best.miou:
            best = metrics
            print(f'[split {split}] new best mIoU {best.miou:.4f}')
            save_metrics(cfg.result_dir, f'split{split}', best)
            if cfg.save_checkpoint:
                save_checkpoint(cfg.snapshot_dir / f'best_split{split}.pth', models,
                                split=split, epoch=epoch, miou=best.miou, config=describe(cfg))
            if pred_dir is not None:
                evaluate(cfg, device, models, shot_images_te, shot_labels_te, query_loader, cfg.max_eval_batches,
                         pred_dir=pred_dir)
    return best


def main():
    print(describe(cfg))
    if cfg.batch_size != 2 * cfg.shot:
        raise ValueError(f'batch_size ({cfg.batch_size}) must equal 2 * shot ({2 * cfg.shot}) for region pasting')
    set_seed(cfg.seed)
    cudnn.benchmark = cfg.cudnn_benchmark
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_tf, test_tf = build_transforms(cfg.crop_size, cfg.ignore_label)
    source_set = SegmentationDataset(cfg.source_root, cfg.source_root / cfg.source_list, cfg.source_img_ext,
                                     cfg.source_label_ext, train_tf, cfg.crop_size)
    source_loader = DataLoader(source_set, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label)
    dice = DiceLoss(cfg.ignore_label)

    models = build_and_init(cfg, device)
    for name, m in zip(('extractor', 'adapter', 'classifier'), models):
        print(f'{name}: {count_parameters(m):,} trainable parameters')

    results = {}
    for split in range(cfg.first_split, cfg.first_split + cfg.num_splits):
        if cfg.reinit_per_split and split != cfg.first_split:
            models = build_and_init(cfg, device)
        best = run_split(cfg, device, models, source_loader, train_tf, test_tf, criterion, dice, split)
        if best is not None:
            results[split] = best

    print_summary(results, 'best mIoU per split')


if __name__ == '__main__':
    main()
