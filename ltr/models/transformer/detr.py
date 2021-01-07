# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Song edited the DETR model
"""
import torch
import torch.nn.functional as F
from torch import nn

import ltr.data.box_ops as box_ops

from .resnet_backbone import build_backbone
from .transformer import build_transformer

class DETR(nn.Module):
    """
        Song : This is the DETR module that performs object tracking

        The differences are :
            1) To fix the query_embed, expecting a tensor [batchsize, N=1, 256],
               the deep feature map from Template branch on ResNet50, [batchsize, C, H, W]
               the QueryProj is used
            2) To remove the num_classes, num_queries, because they are 1
            3) To replace the class_embed to conf_embed ??? to predict the confidence?
    """
    def __init__(self, backbone, transformer, postprocessors):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
        """
        super().__init__()

        self.transformer = transformer
        hidden_dim = transformer.d_model                                        # 256
        self.conf_embed = MLP(hidden_dim, hidden_dim, 1, 3)                     # Song added
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_proj = QueryProj(backbone.num_channels, hidden_dim)          # Song added, project Template features into query
        self.backbone = backbone
        self.postprocessors = postprocessors

    def forward(self, search_imgs, template_imgs):
        """ The forward expects a NestedTensor, which consists of:
               - template_imgs : batched images, of shape [batch_size x 3 x H x W], which are the template branch
               # - template_bb   : batched bounding boxes, of shape [batch_size x 4]

               - search_imgs : batched test images , of shape [batch_size x 3 x H x W], which are the test branch
                               is a cropped squred images, they have the same shape

            It returns a dict with the following elements:

               - "pred_conf": the confidence values for the predicted bboxes

               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
        """
        if len(search_imgs.shape) == 5:
            num_train_imgs, batch_size, C, H, W = search_imgs.shape
            search_imgs = search_imgs.view(num_train_imgs*batch_size, C, H, W).clone().requires_grad_()

            num_test_imgs, batch_size, C, H, W = template_imgs.shape
            template_imgs = template_imgs.view(num_test_imgs*batch_size, C, H, W).clone().requires_grad_()

        target_sizes = torch.unsqueeze(torch.tensor(search_imgs.shape[-2:]), 0) # [1 x 2]
        target_sizes = target_sizes.repeat(search_imgs.shape[0], 1)             # [batch_size x 2]
        target_sizes = target_sizes.cuda()

        search_features, search_pos = self.backbone(search_imgs)
        search_mask = None # Song : we don't use the mask yet

        template_features, template_pos = self.backbone(template_imgs)

        hs = self.transformer(self.input_proj(search_features), search_mask, self.query_proj(template_features), search_pos)[0]
        # print('Song in detr.py: hs.shape = ', hs.shape)                       # [6, batch, 1, 256 ] ??
        outputs_conf = self.conf_embed(hs)                                      # Song added
        outputs_coord = self.bbox_embed(hs).sigmoid()                           # Song why do not output the [x,y,w,h] directly ? they are same

        out = {'pred_conf': outputs_conf[-1], 'pred_boxes': outputs_coord[-1]}

        out = self.postprocessors(out, target_sizes) # [x,y,w,h]

        return out

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api
        Song : ? for tracking , we need to output the box of xywh and confidence
    """
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        scores, boxes = outputs['pred_conf'], outputs['pred_boxes']
        # print('song len(scores), len(out_bbox) : ', len(scores), len(out_bbox))
        assert len(scores) == len(target_sizes)
        # assert target_sizes.shape[1] == 2

        # boxes = box_ops.box_cxcywh_to_xywh(out_bbox) # Song : assume that, the network predict (x, y, w, h)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)                                   # img_h : [batch_size, ], img_w : [batch_size, ]
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # scores = torch.clamp(scores, min=0.0, max=1.0) # Song Not differentiable at 0.0 and 1.0

        batch_size = target_sizes.shape[0]
        boxes = boxes.view(batch_size, -1).clone()
        scores = scores.view(batch_size).clone()

        results = {'pred_conf': scores, 'pred_boxes': boxes}

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class QueryProj(nn.Module):
    """
    Song : project the template features into object query features
    [batch, C=512, H, W] -> [batch, N=1, output_dim=256]

    we can use the ResNet50+PrROIPooling instead

    Probelm, how to stack images with different shapes ? since the bbox are different ?
        - the naive way, to resize + pooling

    Song : If we resize the cropped template images, we make fix the number of the dimmensions, and remove the AdaptiveAvgPool2d ???
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        pool_size = 4*4
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1) # C=512 -> 256
        # self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.fc = nn.Linear(pool_size*output_dim, output_dim)

    def forward(self, x):
        x = self.conv(x)              # [batch, 512, 4, 4] - > [batch, 256, H, W]
        # print('Song in detr.py: x.shape = ', x.shape)
        # x = self.pool(x)              # [batch, 256, H, W] - > [batch, 256, 5, 5]
        x = x.flatten(1)              # [batch, 256, 5, 5] - > [batch, 256*5*5]
        x = F.relu(self.fc(x))        # [batch, 256*5*5] -> [batch, 256]
        # return torch.unsqueeze(x, 1)  # [batch, 1, 256]
        return x.unsqueeze(1)

def build_tracker(backbone_name='resnet50',
                  backbone_pretrained=True,
                  hidden_dim=256,
                  position_embedding='learned',
                  # lr_backbone=1e-5,
                  # masks=False,
                  # dilation=False,
                  dropout=0.1,
                  nheads=8,
                  dim_feedforward=2048,
                  enc_layers=6,
                  dec_layers=6,
                  pre_norm=False):
    '''
    the settings for the transformer :
        -hiddent_dim     : Size of the embeddings (dimension of the transformer)
        -dropout         : Dropout applied in the transformer
        -nheads          : Number of attention heads inside the transformer's attentions
        -dim_feedforward : Intermediate size of the feedforward layers in the transformer blocks
        -enc_layers      : Number of encoding layers in the transformer
        -dec_layers      : Number of decoding layers in the transformer
        -pre_norm        : normalization or not in the transformer (Song guesses)


    the settings for the convolutional backbone :
        -backbone           : Name of the convolutional backbone to use
        # -masks              : Train segmentation head if the flag is provided
        # -dilation           : If true, we replace stride with dilation in the last convolutional block (DC5)
        -position_embedding : Type of positional embedding to use on top of the image features, chices=('sine', 'learned')
    '''

    # backbone = build_backbone(backbone=backbone,
    #                           backbone_pretrained=True,
    #                           hidden_dim=hidden_dim,
    #                           position_embedding=position_embedding,
    #                           lr_backbone=lr_backbone,
    #                           masks=masks,
    #                           dilation=dilation)
    backbone = build_backbone(backbone_name=backbone_name,
                              backbone_pretrained=backbone_pretrained,
                              hidden_dim=hidden_dim,
                              position_embedding=position_embedding)

    transformer = build_transformer(hidden_dim=hidden_dim,
                                    dropout=dropout,
                                    nheads=nheads,
                                    dim_feedforward=dim_feedforward,
                                    enc_layers=enc_layers,
                                    dec_layers=dec_layers,
                                    pre_norm=pre_norm)
    # Convert the [cx, cy, h, w] into [x, y, h, w]
    postprocessors = PostProcess()

    model = DETR(backbone, transformer, postprocessors)

    return model
