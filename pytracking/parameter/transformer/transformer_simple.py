from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone
from pytracking.utils.loading import load_network
import torch

from ltr.models.transformer.detr_simple import build_tracker
def parameters():

    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    # params.seg_to_bb_mode = 'var'
    # params.max_scale_change = (0.95, 1.1)
    # params.min_mask_area = 100

    params.use_gpu = True

    # params.image_sample_size = (30 * 16, 52 * 16) # [480, 832], (288, 288)
    params.search_area_scale = 5.0
    params.border_mode = 'replicate'
    # params.patch_max_scale_change = None
    params.output_sz = (288, 288) # torch.Tensor([288, 288])

    # Learning parameters
    # params.sample_memory_size = 32
    # params.learning_rate = 0.2
    # params.init_samples_minimum_weight = 0
    # params.train_skipping = 5

    # Net optimization params
    # params.update_target_model = True
    # params.net_opt_iter = 20
    # params.net_opt_update_iter = 5

    mean = (0.485, 0.456, 0.406)
    params._mean = torch.Tensor(mean).view(1, -1, 1, 1)
    std = (0.229, 0.224, 0.225)
    params._std = torch.Tensor(std).view(1, -1, 1, 1)

    # params.init_with_box = True
    # params.lower_init_weight = True

    # params.net = NetWithBackbone(net_path=,
    #                              use_gpu=params.use_gpu,
    #                              mean=(0.485, 0.456, 0.406),
    #                              std=(0.229, 0.224, 0.225))

    # params.net = load_network('DETR_SIMPLE.pth.tar')
    print('Build tracker ...')
    params.net = build_tracker(backbone_name='resnet50',
                               output_layers=['layer3'],
                               num_channels=1024,                       # 2048 for layer4, 1024 for layer3
                               backbone_pretrained=True,
                               hidden_dim=256,
                               position_embedding='learned',            # position_embedding='sine',
                               dropout=0.1,
                               nheads=8,
                               dim_feedforward=2048,
                               enc_layers=6,
                               dec_layers=6,
                               pre_norm=False)
    # # If load a part of the pretrained_dct
    # pretrained_dict = torch.load('/home/yans/pytracking-models/pytracking/networks/DETR_SIMPLE.pth.tar', map_location='cpu')
    # model_dict = model.state_dict()
    #
    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # # 3. load the new state dict
    # model.load_state_dict(pretrained_dict)
    print('Loading models for DETR(Simple version) ...')
    checkpoint_dict = torch.load('/home/yans/pytracking-models/pytracking/networks/DETR_SIMPLE.pth.tar', map_location='cpu')
    params.net.load_state_dict(checkpoint_dict['net'])
    params.net.eval()

    # params.vot_anno_conversion_type = 'preserve_area'
    return params
