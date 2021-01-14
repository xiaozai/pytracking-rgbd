from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import numpy as np
import math
import time
from pytracking.features.preprocessing import numpy_to_torch, sample_patch
import cv2
from ltr.data.processing_utils import sample_target, transform_image_to_crop, crop_and_resize


class Transformer_Simple(BaseTracker):
    '''
         the DETR_SIMPLE  network,
            - initialize(template, info) -> crop and resize template images
            - track(search_imgs, template_imgs, template_bb) -> xywh
    '''
    def initialize(self, template, info: dict) -> dict:
        print('Start to initialize the Transformer_Simple ....')
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Time initialization
        tic = time.time()
        # Output
        out = {}

        self.net = self.params.net

        # Convert template image, using the 1st image as the Template
        state = info['init_bbox']
        self.prev_box = torch.Tensor(state)
        crop_center = torch.Tensor([self.prev_box[1] + (self.prev_box[3] - 1)/2, self.prev_box[0] + (self.prev_box[2] - 1)/2])
        search_sz = torch.Tensor([self.params.search_area_scale * self.prev_box[3], self.params.search_area_scale * self.prev_box[2]])

        crop_box = [crop_center[0] - search_sz[0] / 2 , crop_center[1] - search_sz[1] / 2, search_sz[0], search_sz[1]] # [x, y, w, h]
        crop_box = [int(cb) for cb in crop_box]

        template_crop, template_bb_crop = crop_and_resize(template, self.prev_box, crop_box, self.params.output_sz)

        self.template = numpy_to_torch(template_crop)                           # [1, 3, H ,W]
        self.template = self.template / 255
        self.template -= self.params._mean
        self.template /= self.params._std

        self.template_bb = torch.Tensor(template_bb_crop).view(1, 4)

        if self.params.use_gpu:
            self.net = self.net.cuda()
            self.template = self.template.cuda()
            self.template_bb = self.template_bb.cuda()

        out['time'] = time.time() - tic

        return out

    def track(self, image, info: dict = None) -> dict:

        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Crop and Resize
        # crop_center = torch.Tensor([self.prev_box[1] + (self.prev_box[3] - 1)/2, self.prev_box[0] + (self.prev_box[2] - 1)/2])
        # search_sz = torch.Tensor([self.params.search_area_scale * self.prev_box[3], self.params.search_area_scale * self.prev_box[2]])
        # crop_box = [crop_center[0] - search_sz[0] / 2, crop_center[1] - search_sz[1] / 2, search_sz[0], search_sz[1]]
        # crop_box = [int(cb) for cb in crop_box]
        # im, _ = crop_and_resize(image, None, crop_box, self.params.output_sz)

        # Global Search
        H, W, C = image.shape
        im = cv2.resize(image, self.params.output_sz)

        im = numpy_to_torch(im)                                                 # [1, C, H, W]
        # Convert image
        im = im / 255
        im -= self.params._mean # Tensor
        im /= self.params._std  # Tensor

        if self.params.use_gpu:
            im = im.cuda()

        pred_box = self.net(im, self.template, self.template_bb)                # xywh
        pred_box = pred_box['pred_boxes'].detach().cpu().numpy()[0]
        # convert the predicted pos back to image coords
        # pred_box = self.back_to_image_coords(pred_box, crop_box, self.params.output_sz)

        # Back to Global Coordinates
        pred_box[0] = pred_box[0] * W / self.params.output_sz[1]
        pred_box[1] = pred_box[1] * H / self.params.output_sz[0]
        pred_box[2] = pred_box[2] * W / self.params.output_sz[1]
        pred_box[3] = pred_box[3] * H / self.params.output_sz[0]

        if pred_box[2] > 10 and pred_box[3] > 10:
            # update
            self.prev_box = pred_box
        out = {'target_bbox': pred_box}

        return out

    def back_to_image_coords(self,pred_box, crop_bb, output_sz):
        rescale_factor = 1.0 * output_sz[0] /  crop_bb[2]
        ori_box = pred_box / rescale_factor
        ori_box[0] += crop_bb[0]
        ori_box[1] += crop_bb[1]

        return ori_box
