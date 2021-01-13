from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import numpy as np
import math
import time
from pytracking.features.preprocessing import numpy_to_torch, sample_patch
import cv2

class Transformer_Simple(BaseTracker):
    '''
    Transformer_Simple, only use the first frames as the template

    only need to overload the functions : initialize and track
    '''
    def initialize(self, template, info: dict) -> dict:
        '''
        Song : To initialize the tracker with the template feature maps

            we just extract the template feature maps for tracking

            our network doesnot require to crop the image ???
        '''
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Song, the DETR_SIMPLE  network, net(search_imgs, template_imgs, template_bb) -> xywh
        self.net = self.params.net

        # Convert template image, using the 1st image as the Template
        self.template = numpy_to_torch(template) # [1, 3, H ,W]

        # Time initialization
        tic = time.time()
        # Output
        out = {}

        # Get target position and size
        state = info['init_bbox']
        # init_mask = info.get('init_mask', None)

        # Set target center and target size
        # self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        # self.target_sz = torch.Tensor([state[3], state[2]])

        self.template_bb = torch.Tensor(state).view(1, 4)

        if self.params.use_gpu:
            self.template = self.template.cuda()
            self.template_bb = self.template_bb.cuda()

        # Set sizes
        # sz = self.params.image_sample_size
        # self.img_sample_sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        # self.img_support_sz = self.img_sample_sz

        # Set search area.
        # search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        # self.target_scale = math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        # self.base_target_sz = self.target_sz / self.target_scale

        # Extract and transform sample
        # self.init_template_feat = self.extract_template_feature(template, self.pos)

        out['time'] = time.time() - tic

        return out

    def track(self, image, info: dict = None) -> dict:
        print('Song in tracker.track (img )')
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        # 1) crop image centered at previous prediction
        # 2) resize cropped image to 288x288

        h, w, c = image.shape
        im = cv2.resize(image, (288, 288))
        im = numpy_to_torch(im) # [1, C, H, W]

        # im = self.net.preprocess_image(im)
        im = im / 255
        im -= self.params._mean
        im /= self.params._std
        if self.params.use_gpu:
            im = im.cuda()
        #
        # if self.params.use_gpu:
        #     im = im.cuda()

        # Crop search region , based on the previous target pos and img_sample_sz
        # remember how to convert the predicted box back to the orginal size
        # search_region, patch_coord = sample_patch(im, self.pos, self.img_sample_sz, self.output_sz)
        pred_pos = self.net(im, self.template, self.template_bb) # xywh
        pred_pos = pred_pos['pred_boxes'].detach().cpu().numpy()[0]
        # convert the predicted pos back to image coords
        # pred_pos = self.convert_to_image_coords(pred_pos, ...)
        # self.pos = pred_pos

        # 3) back to the image Coordinates 
        pred_pos = pred_pos * h  / 288.0

        out = {'target_bbox': pred_pos}

        return out
