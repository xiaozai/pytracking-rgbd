import torch
import torch.nn as nn
import torch.optim as optim
from ltr.dataset import Lasot, Got10k, TrackingNet, MSCOCOSeq
from ltr.data import processing, sampler, LTRLoader
from ltr.models.tracking import dimpnet
from ltr.models.transformer import detr
import ltr.models.loss as ltr_losses
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as tfm
from ltr import MultiGPU
from ltr.train_settings.transformer.criterion import SetCriterion

def run(settings):
    settings.description = 'Default train settings for Transformer with ResNet50 as backbone.'
    settings.batch_size = 32
    settings.num_workers = 8
    settings.multi_gpu = False
    settings.print_interval = 1
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = 5.0
    # settings.output_sigma_factor = 1/4            # Song ??
    # settings.target_filter_sz = 4                 # Song ???
    # settings.feature_sz = 18
    # settings.output_sz = settings.feature_sz * 16 # Song 18*16 = 288 , so the test images crops are 288*288
    settings.output_sz = 288
    settings.center_jitter_factor = {'train': 3, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0.25, 'test': 0.5}
    # settings.hinge_threshold = 0.05
    # settings.print_stats = ['Loss/total', 'Loss/iou', 'ClfTrain/clf_ce', 'ClfTrain/test_loss']

    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    # trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))
    coco_train = MSCOCOSeq(settings.env.coco_dir)

    # Validation datasets
    got10k_val = Got10k(settings.env.got10k_dir, split='votval')


    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    # Data Processors
    data_processing_train = processing.TransformerProcessing(search_area_factor=settings.search_area_factor,
                                                             output_sz=settings.output_sz,
                                                             center_jitter_factor=settings.center_jitter_factor,
                                                             scale_jitter_factor=settings.scale_jitter_factor,
                                                             mode='pair', # 'sequence',
                                                             transform=transform_train,
                                                             joint_transform=transform_joint,
                                                             template_output_sz=(112, 112)) # crop and resize template images with bbox

    data_processing_val = processing.TransformerProcessing(search_area_factor=settings.search_area_factor,
                                                           output_sz=settings.output_sz,
                                                           center_jitter_factor=settings.center_jitter_factor,
                                                           scale_jitter_factor=settings.scale_jitter_factor,
                                                           mode='pair', # 'sequence',
                                                           transform=transform_val,
                                                           joint_transform=transform_joint,
                                                           template_output_sz=(112, 112)) # crop and resize template images with bbox

    # Train sampler and loader # Song, ignore Tracking-net currently
    dataset_train = sampler.TransformerSampler([lasot_train, got10k_train, coco_train], [0.25,1,1],
                                        samples_per_epoch=26000, max_gap=30, num_test_frames=1, num_train_frames=1,
                                        processing=data_processing_train)

    # loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
    #                          shuffle=True, drop_last=True, stack_dim=1)
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=0)

    # Validation samplers and loaders
    dataset_val = sampler.TransformerSampler([got10k_val], [1], samples_per_epoch=5000, max_gap=30,
                                      num_test_frames=1, num_train_frames=1,
                                      processing=data_processing_val)

    # loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
    #                        shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=0)

    # Create network and actor
    net = detr.build_tracker(backbone_name='resnet50',
                             backbone_pretrained=True,
                             hidden_dim=256,
                             position_embedding='learned', # position_embedding='sine',
                             # lr_backbone=1e-5,
                             # masks=False,
                             # dilation=False,
                             dropout=0.1,
                             nheads=8,
                             dim_feedforward=2048,
                             enc_layers=6,
                             dec_layers=6,
                             pre_norm=False)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        net = MultiGPU(net, dim=1)

    objective = SetCriterion(losses=['boxes_iou_conf'])

    loss_weight = {'bbox': 1, 'iou': 100, 'conf': 100}

    actor = actors.TransformerActor(net=net, objective=objective, loss_weight=loss_weight)

    # Optimizer, transformer + pos embedding (backbone.0 is the ResNet, backbone.1 is the pos embedding)
    param_dicts = [
        {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in net.named_parameters() if "backbone.1" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 200)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    trainer.train(200, load_latest=True, fail_safe=True)
