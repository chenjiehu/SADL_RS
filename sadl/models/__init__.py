from .extractor import FeatureExtractor, SuperpixelBatch, pool_superpixels
from .modules import SelfAttention, ResidualFeatureAdapter
from .propagation import LabelPropagation, SuperpixelClassifier, class_prototypes
from .encoder import FeatureEncoder, ASPPHead
from .resnet import DilatedResNet, resnet50_backbone


def build_models(cfg, device):
    extractor = FeatureExtractor(cfg.pretrained_backbone, cfg.feature_dim, cfg.aspp_dilations).to(device)
    adapter = ResidualFeatureAdapter(cfg.feature_dim).to(device)
    propagation = LabelPropagation(
        feature_dim=cfg.feature_dim,
        alpha=cfg.alpha,
        learn_alpha=cfg.learn_alpha,
        edge_threshold=cfg.edge_threshold,
        confidence_threshold=cfg.confidence_threshold,
        ignore_label=cfg.ignore_label,
    )
    classifier = SuperpixelClassifier(propagation, cfg.ignore_label).to(device)
    return extractor, adapter, classifier


__all__ = [
    'FeatureExtractor', 'SuperpixelBatch', 'pool_superpixels',
    'SelfAttention', 'ResidualFeatureAdapter',
    'LabelPropagation', 'SuperpixelClassifier', 'class_prototypes',
    'FeatureEncoder', 'ASPPHead', 'DilatedResNet', 'resnet50_backbone',
    'build_models',
]
