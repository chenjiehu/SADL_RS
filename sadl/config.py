import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    gpu: str = '0'
    seed: int = 3
    num_workers: int = 1
    cudnn_benchmark: bool = True

    source_root: Path = PROJECT_ROOT / 'dataset' / 'WHDLD'
    source_list: str = 'image_list.txt'
    source_img_ext: str = '.jpg'
    source_label_ext: str = '.png'
    source_classes: int = 6

    target_root: Path = PROJECT_ROOT / 'dataset' / 'GID5_new'
    target_img_ext: str = '.tif'
    target_label_ext: str = '.tif'
    target_classes: int = 5
    shot_dir: str = 'shot_GID5_new'
    query_dir: str = 'query_GID5_new'

    pretrained_backbone: Path = PROJECT_ROOT / 'pretrained_model' / 'resnet50-19c8e357.pth'
    result_dir: Path = PROJECT_ROOT / 'result'
    snapshot_dir: Path = PROJECT_ROOT / 'snapshots' / 'SADL'
    save_checkpoint: bool = True
    save_pred: bool = False

    pretrain_epochs: int = 20
    pretrain_iters_per_epoch: int = 0
    pretrain_lr: float = 1e-4
    source_val_list: str = ''
    pretrain_checkpoint: Path = PROJECT_ROOT / 'snapshots' / 'SADL' / 'pretrain.pth'
    init_from: str = str(PROJECT_ROOT / 'snapshots' / 'SADL' / 'pretrain.pth')

    shot: int = 5
    first_split: int = 0
    num_splits: int = 5
    reinit_per_split: bool = True
    epochs: int = 10
    iters_per_epoch: int = 30
    max_eval_batches: int = 200
    batch_size: int = 10
    crop_size: int = 256
    ignore_label: int = 255

    lr: float = 5e-5
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_milestones: Sequence[int] = (10000, 20000, 30000)
    lr_gamma: float = 0.1
    freeze_bn: bool = True

    feature_dim: int = 64
    aspp_dilations: Sequence[int] = (6, 12)
    alpha: float = 0.99
    learn_alpha: bool = False
    edge_threshold: float = 0.9
    confidence_threshold: float = 1.0

    n_segments: int = 60
    slic_compactness: float = 20.0
    random_conv_prob: float = 0.8
    random_conv_kernels: Sequence[int] = (1, 3, 5, 7, 11, 15)
    max_phase_steps: int = 3
    max_phase_lr: float = 1e-4
    noise_std: float = 1e-4

    target_label_remap: Sequence[int] = (1, 6, 7, 5, 8)
    replace_categories: Sequence[int] = (0, 2, 3)
    common_labels_source: Sequence[int] = (1, 5)
    common_labels_target: Sequence[int] = (1, 5)
    domain_loss_weight: float = 0.5
    domain_min_samples: int = 10
    min_shot_classes_train: int = 3
    use_adapter_in_eval: bool = True

    @property
    def joint_classes(self) -> int:
        return max(self.source_classes - 1, max(self.target_label_remap)) + 1

    @property
    def replace_values(self) -> tuple:
        return tuple(self.target_label_remap[c] for c in self.replace_categories)


def _str2bool(value: str) -> bool:
    return value.lower() in ('1', 'true', 'yes', 'y')


def _add_argument(parser: argparse.ArgumentParser, name: str, default) -> None:
    flag = '--' + name.replace('_', '-')
    if isinstance(default, bool):
        parser.add_argument(flag, type=_str2bool, default=default)
    elif isinstance(default, (tuple, list)):
        elem_type = type(default[0]) if default else int
        parser.add_argument(flag, type=lambda s, t=elem_type: tuple(t(x) for x in s.split(',')), default=tuple(default))
    elif isinstance(default, Path):
        parser.add_argument(flag, type=Path, default=default)
    else:
        parser.add_argument(flag, type=type(default), default=default)


def parse_config(argv=None) -> Config:
    parser = argparse.ArgumentParser(description='SADL training')
    for f in fields(Config):
        _add_argument(parser, f.name, f.default)
    return Config(**vars(parser.parse_args(argv)))


def describe(cfg: Config) -> str:
    lines = [f'{f.name}: {getattr(cfg, f.name)}' for f in fields(Config)]
    lines.append(f'joint_classes: {cfg.joint_classes}')
    lines.append(f'replace_values: {cfg.replace_values}')
    return '\n'.join(lines)
