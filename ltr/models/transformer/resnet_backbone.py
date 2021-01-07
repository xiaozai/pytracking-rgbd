# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

# from .misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import ltr.models.backbone as backbones

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, num_channels=2048):
        self.num_channels = num_channels
        super().__init__(backbone, position_embedding)

    def forward(self, x):
        xs = self[0](x)                               # resnet50 features
        out, pos = [], []
        for name, feat in xs.items():
            out.append(feat)                          # # features from ResNet50 layer4 : [batch x 2048 x 9 x 9], [batch x 1024 x 18 x 18] for layer 3
            pos.append(self[1](feat).to(feat.dtype))  # # position encoding

        return out[-1], pos[-1]

def build_backbone(backbone_name='resnet50', output_layers=['layer4'], num_channels=2048, backbone_pretrained=True, hidden_dim=256, position_embedding='learned'):

    position_embedding = build_position_encoding(hidden_dim, position_embedding=position_embedding)

    if backbone_name == 'resnet18':
        # num_channels = 512
        backbone = backbones.resnet18(output_layers=output_layers, pretrained=backbone_pretrained)
    else:
        # num_channels = 1024 # for layer3, 2048 for layer4
        backbone = backbones.resnet50(output_layers=output_layers, pretrained=backbone_pretrained)

    model = Joiner(backbone, position_embedding, num_channels=num_channels)

    return model

def build_backbone_pos(backbone_name='resnet50', output_layers=['layer4'], num_channels=2048, backbone_pretrained=True, hidden_dim=256, position_embedding='learned'):

    position_embedding = build_position_encoding(hidden_dim, position_embedding=position_embedding)

    if backbone_name == 'resnet18':
        # num_channels = 512
        backbone = backbones.resnet18(output_layers=output_layers, pretrained=backbone_pretrained)
    else:
        # num_channels = 1024 # for layer3, 2048 for layer4
        backbone = backbones.resnet50(output_layers=output_layers, pretrained=backbone_pretrained)

    model = Joiner(backbone, position_embedding, num_channels=num_channels)

    return model


class template_backbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        xs = self.backbone(x)                               # resnet50 features
        out = []
        for name, feat in xs.items():
            out.append(feat)                          # # features from ResNet50 layer4 : [batch x 2048 x 9 x 9], [batch x 1024 x 18 x 18] for layer 3

        return out[-1]

def build_template_backbone(backbone_name='resnet50', output_layers=['layer4'], backbone_pretrained=True):

    if backbone_name == 'resnet18':
        # num_channels = 512
        backbone = backbones.resnet18(output_layers=output_layers, pretrained=backbone_pretrained)
    else:
        # num_channels = 1024 # for layer3, 2048 for layer4
        backbone = backbones.resnet50(output_layers=output_layers, pretrained=backbone_pretrained)

    model = template_backbone(backbone)
    return model

def build_resnet_backbone(backbone_name='resnet50', output_layers=['layer4'], backbone_pretrained=True):

    if backbone_name == 'resnet18':
        # num_channels = 512
        backbone = backbones.resnet18(output_layers=output_layers, pretrained=backbone_pretrained)
    else:
        # num_channels = 1024 # for layer3, 2048 for layer4
        backbone = backbones.resnet50(output_layers=output_layers, pretrained=backbone_pretrained)

    model = template_backbone(backbone)
    return model
