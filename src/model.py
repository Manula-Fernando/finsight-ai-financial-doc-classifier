from __future__ import annotations
import torch
import torch.nn as nn
import timm

from . import config


class DocClassifier(nn.Module):
    def __init__(self, num_classes: int, backbone_name: str = None, dropout: float = None):
        super().__init__()
        backbone_name = backbone_name or config.BACKBONE
        dropout = config.DROPOUT if dropout is None else dropout
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool='avg')
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits
