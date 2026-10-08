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

from sadl.data.datasets import SegmentationDataset
from sadl.data.superpixel import refined_slic
from sadl.engine import forward_episode, save_checkpoint
from sadl.losses import DiceLoss
from sadl.metrics import confusion_matrix, summarize
from sadl.models import build_models
from sadl.utils import Averager, count_parameters, set_seed, set_train_mode

torch.multiprocessing.set_sharing_strategy('file_system')


def split_episode(images, labels, shot):
    return images[:shot], labels[:shot], images[shot:], labels[shot:]


def run_epoch(cfg, device, models, loader, criterion, dice, optimizer=None, scheduler=None, max_iters=0):
    training = optimizer is not None
    if training:
        set_train_mode(models, cfg.freeze_bn)
    else:
        for m in models:
            m.eval()

    num_classes = cfg.source_classes
    loss_meter = Averager()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    for it, (images, labels, _) in enumerate(loader):
        if max_iters and it >= max_iters:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        shot_i, shot_l, query_i, query_l = split_episode(images, labels, cfg.shot)
        if torch.unique(shot_l).numel() < cfg.min_shot_classes_train:
            continue

        with torch.set_grad_enabled(training):
            segments = refined_slic(images, labels, cfg.n_segments, cfg.slic_compactness)
            pred, _ = forward_episode(models, shot_i, shot_l, query_i, query_l, segments, num_classes)
            loss = criterion(pred, query_l) + dice(pred, query_l)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

        loss_meter.add(loss.item())
        confusion += confusion_matrix(pred.argmax(1), query_l, num_classes, cfg.ignore_label)

    return loss_meter.item(), summarize(confusion.cpu().numpy())


def main():
    print(describe(cfg))
    if cfg.batch_size <= cfg.shot:
        raise ValueError(f'batch_size ({cfg.batch_size}) must be larger than shot ({cfg.shot})')
    set_seed(cfg.seed)
    cudnn.benchmark = cfg.cudnn_benchmark
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from sadl.data.transforms import build_transforms
    train_tf, test_tf = build_transforms(cfg.crop_size, cfg.ignore_label)

    train_set = SegmentationDataset(cfg.source_root, cfg.source_root / cfg.source_list, cfg.source_img_ext,
                                    cfg.source_label_ext, train_tf, cfg.crop_size)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    val_loader = None
    if cfg.source_val_list:
        val_set = SegmentationDataset(cfg.source_root, cfg.source_root / cfg.source_val_list, cfg.source_img_ext,
                                      cfg.source_label_ext, test_tf, cfg.crop_size)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label)
    dice = DiceLoss(cfg.ignore_label)

    models = build_models(cfg, device)
    for name, m in zip(('extractor', 'adapter', 'classifier'), models):
        print(f'{name}: {count_parameters(m):,} trainable parameters')

    params = [p for m in models for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.pretrain_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=list(cfg.lr_milestones), gamma=cfg.lr_gamma)

    best_miou = -1.0
    ckpt_dir = cfg.pretrain_checkpoint.parent
    for epoch in range(cfg.pretrain_epochs):
        train_loss, train_metrics = run_epoch(cfg, device, models, train_loader, criterion, dice,
                                             optimizer, scheduler, cfg.pretrain_iters_per_epoch)
        print(f'[pretrain epoch {epoch}] loss {train_loss:.4f}\n{train_metrics}')

        if val_loader is not None:
            val_loss, val_metrics = run_epoch(cfg, device, models, val_loader, criterion, dice)
            print(f'[pretrain epoch {epoch}] val loss {val_loss:.4f}\n{val_metrics}')
            score = val_metrics.miou
        else:
            score = train_metrics.miou

        save_checkpoint(ckpt_dir / f'pretrain_epoch{epoch}.pth', models, epoch=epoch, miou=score)
        if score > best_miou:
            best_miou = score
            save_checkpoint(cfg.pretrain_checkpoint, models, epoch=epoch, miou=best_miou)

    print(f'pretraining finished, best mIoU {best_miou:.4f}')
    print(f'  best checkpoint:      {cfg.pretrain_checkpoint}')
    print(f'  per-epoch checkpoints: {ckpt_dir}/pretrain_epoch{{0..{cfg.pretrain_epochs - 1}}}.pth')


if __name__ == '__main__':
    main()
