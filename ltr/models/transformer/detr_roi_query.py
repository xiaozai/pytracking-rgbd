# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Song edited the DETR model
"""
import torch
import torch.nn.functional as F
from torch import nn

import ltr.data.box_ops as box_ops

from .resnet_backbone import build_backbone_pos, build_resnet_backbone
from .transformer import build_transformer_query
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D

class DETR_ROI_QUERY(nn.Module):
    """
        Song : This is the DETR module that performs object tracking

        The differences are :
            1) To fix the query_embed, expecting a tensor [batchsize, N=1, 256],
               the deep feature map from Template branch on ResNet50, [batchsize, C, H, W]
               the QueryProj is used
            2) To remove the num_classes, num_queries, because they are 1
            3) To replace the class_embed to conf_embed ??? to predict the confidence?
    """
    def __init__(self, search_backbone_pos, transformer, template_backbone, query_encoder_size=9, roi_windowsize=3, prroipool_scalefactor=1/32):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            postprocessors : post-processing
        """
        super().__init__()

        self.transformer = transformer
        hidden_dim = transformer.d_model
        # self.conf_embed = MLP(hidden_dim, hidden_dim, 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # Project the channels to hidden_dim
        self.input_proj = nn.Conv2d(search_backbone_pos.num_channels, hidden_dim, kernel_size=1)
        # PrROIPooling layer for template + bbox
        self.prroi_pool = PrRoIPool2D(roi_windowsize, roi_windowsize, prroipool_scalefactor)
        # Project the template roi feature map into Query Embedding for Decoder
        self.query_proj_decoder = QueryProj_Decoder(search_backbone_pos.num_channels, hidden_dim, roi_windowsize)
        # Project the template roi feature map into Query Embedding for Encoder
        self.query_proj_encoder = QueryProj_Encoder(search_backbone_pos.num_channels, hidden_dim, query_encoder_size) # Song  assume that the ResNet50 - layer4

        self.search_backbone_pos = search_backbone_pos
        self.template_backbone = template_backbone

    def forward(self, search_imgs, template_imgs, template_bb):
        """ The forward expects a NestedTensor, which consists of:
               - template_imgs : batched images, of shape [batch_size x 3 x H x W], which are the template branch
               - template_bb   : batched bounding boxes, of shape [batch_size x 4], (x,y,w,h)

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
            '''
            Song : in the processing ,
                    if the mode is the "sequence", it will return [num_test_imgs, batch_size, 3, H, W]
                    if the mode is the "pair", it will return [batch_size, 3, H, W]
            '''
            num_train_imgs, batch_size, C, H, W = search_imgs.shape
            search_imgs = search_imgs.view(num_train_imgs*batch_size, C, H, W).clone().requires_grad_()

            num_test_imgs, batch_size, C, H, W = template_imgs.shape
            template_imgs = template_imgs.view(num_test_imgs*batch_size, C, H, W).clone().requires_grad_()

        target_sizes = torch.unsqueeze(torch.tensor(search_imgs.shape[-2:]), 0) # [1 x 2], [288x288]
        target_sizes = target_sizes.repeat(search_imgs.shape[0], 1)             # [batch_size x 2]
        target_sizes = target_sizes.cuda()

        search_features, search_pos = self.search_backbone_pos(search_imgs)
        search_mask = None # Song : we don't use the mask yet

        # Template features from T-1 frame
        template_features = self.template_backbone(template_imgs)               # [batch, 2048, 9, 9] for layer4, [batch, 1024, 18, 18] for layer3
        # PrROIPooling on template branch Add batch_index to rois, bb is [batch, 4]
        batch_size = template_bb.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(template_bb.device)
        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb = template_bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi_bb = torch.cat((batch_index, bb), dim=1)
        template_roi = self.prroi_pool(template_features, roi_bb)               # [batch, 2048, roi_windowsize, roi_windowsize]

        # Transformer
        hs = self.transformer(self.input_proj(search_features),
                              search_mask,
                              self.query_proj_encoder(template_roi),
                              self.query_proj_decoder(template_roi),
                              search_pos)[0]

        # outputs_conf = self.conf_embed(hs)                                    # Song added
        outputs_coord = self.bbox_embed(hs).sigmoid()                           # Song why do not output the [x,y,w,h] directly ? they are same

        # out = {'pred_conf': outputs_conf[-1], 'pred_boxes': outputs_coord[-1]}
        # out = {'pred_boxes': outputs_coord[-1]}                                 # can be [x, y, w, h]
        # out = self.postprocessors(out, target_sizes) # [x,y,w,h]

        # Post-processing, to scale the output bbox
        boxes = box_ops.box_cxcywh_to_xywh(outputs_coord[-1])                   # force the network to output xywh
        img_h, img_w = target_sizes.unbind(1)                                   # img_h : [batch_size, ], img_w : [batch_size, ]
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]                                   # back to original size in 288x288

        batch_size = target_sizes.shape[0]
        boxes = boxes.view(batch_size, -1).clone().requires_grad_()

        out = {'pred_boxes': boxes}

        return out

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

class QueryProj_Decoder(nn.Module):
    """
    Song : project the template features into object query features
    [batch, C=512, H, W] -> [batch, N=1, output_dim=256]

    Song borrows the ideas from the paper : Video Actiion Transformer Network, this procedure as HighRes query preprocessing
    1) to reduce the dimensinality by a 1x1 convolution
    2) to concatenate the cells of the resulting feature map into a vector
    3) to reduce the dimensionality of this feature mpa using a linear layer to 256D
    """
    def __init__(self, input_dim, output_dim, feature_size):
        super().__init__()

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1)             # C=512 -> 256
        self.fc = nn.Linear(feature_size*feature_size*output_dim, output_dim)

    def forward(self, x):
        x = self.conv(x)                                                        # [batch, 512, H, W] - > [batch, 256, H, W]
        x = x.flatten(1)                                                        # [batch, 256, 5, 5] - > [batch, 256*5*5]
        x = F.relu(self.fc(x))                                                  # [batch, 256*5*5]   -> [batch, 256]
        return x.unsqueeze(1)                                                   # [batch, 1, 256]
        # return torch.unsqueeze(x, 1)

class QueryProj_Encoder(nn.Module):
    """
    Song : project the template features into object query features
    Template ROI [batch, 2048, 3, 3] ->  the same shape as the src [batch, 256, 9, 9]
    """
    def __init__(self, input_dim, output_dim, output_size):
        super().__init__()

        self.output_size = output_size
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1)
    def forward(self, x):

        x = self.conv(x)                                                        # [batch x 2048 x 3 x 3] --> [batch x 256 x 3 x 3]
        x = nn.functional.interpolate(x, size=self.output_size)
        return x

def build_tracker(backbone_name='resnet50',
                  output_layers=['layer4'],
                  num_channels=2048, # for layer4, and 1024 for layer3
                  backbone_pretrained=True,
                  hidden_dim=256,
                  position_embedding='learned',
                  dropout=0.1,
                  nheads=8,
                  dim_feedforward=2048,
                  enc_layers=6,
                  dec_layers=6,
                  pre_norm=False,
                  roi_windowsize=3):
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
        -backbone             : Name of the convolutional backbone to use
        # -masks              : Train segmentation head if the flag is provided
        # -dilation           : If true, we replace stride with dilation in the last convolutional block (DC5)
        -position_embedding   : Type of positional embedding to use on top of the image features, chices=('sine', 'learned')

        -roi_windowsize       : the window size for PrROIPooling layer on the template branchs
    '''
    # Positional Embedding + Resnet50
    search_backbone_pos = build_backbone_pos(backbone_name=backbone_name,
                                             output_layers=output_layers,
                                             num_channels=num_channels,
                                             backbone_pretrained=backbone_pretrained,
                                             hidden_dim=hidden_dim,
                                             position_embedding=position_embedding)

    # Single ResNet backbone, for template branch
    template_backbone = build_resnet_backbone(backbone_name=backbone_name,
                                              output_layers=output_layers,
                                              backbone_pretrained=backbone_pretrained)

    transformer = build_transformer_query(hidden_dim=hidden_dim,
                                          dropout=dropout,
                                          nheads=nheads,
                                          dim_feedforward=dim_feedforward,
                                          enc_layers=enc_layers,
                                          dec_layers=dec_layers,
                                          pre_norm=pre_norm)


    '''
        ResNet50 - layer4 : batchx2048x9x9 ,  scale_factor = 1/32 , 288x288 -> 9x9
        ResNet50 - layer3 : batchx1024x18x18, scale_factor = 1/16,  288x288 -> 18x18
    '''
    if 'layer4' in output_layers:
        scale_factor = 1/32
        query_encoder_size = 9
    elif 'layer3' in output_layers:
        scale_factor = 1/16
        query_encoder_size = 18

    model = DETR_ROI_QUERY(search_backbone_pos, transformer, template_backbone,
                           query_encoder_size=query_encoder_size,
                           roi_windowsize=roi_windowsize,
                           prroipool_scalefactor=scale_factor) # postprocessors)

    return model
