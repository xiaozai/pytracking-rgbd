class EnvironmentSettings:
    def __init__(self):
        # self.workspace_dir = '/home/yan/Projects/pytracking/ltr/train_atom/'    # Base directory for saving network checkpoints.
        self.workspace_dir = '/home/yans/pytracking-models/ltr/'
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.lasot_dir = '/home/yans/Datasets/LaSOTBenchmark/'
        self.got10k_dir = '/home/yans/Datasets/Got10k/train/'
        self.trackingnet_dir = '/home/yans/Datasets/TrackingNet/'
        self.coco_dir = '/home/yans/Datasets/COCO/'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.cdtb_dir = '/home/yans/Datasets/CDTB/'
