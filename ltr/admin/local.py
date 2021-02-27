class EnvironmentSettings:
    def __init__(self):
        # self.workspace_dir = '/home/yan/Projects/pytracking/ltr/train_atom/'    # Base directory for saving network checkpoints.

        self.workspace_dir = '/home/yan/Data2/pytracking-models/ltr/'
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'

        root_path = '/home/yan/Data2/Datasets/'

        self.got10k_dir = root_path + 'Got10k/train/'
        self.trackingnet_dir = root_path + 'TrackingNet/'
        self.coco_dir = root_path + 'COCO/'
        self.lasot_dir = root_path + 'LaSOTBenchmark/'

        self.got10kdepth_dir = root_path + 'EstimatedDepth/Got10k_densedepth/train/'
        self.trackingnetdepth_dir = root_path + 'EstimatedDepth/TrackingNet_densedepth/'


        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''




        self.lasotdepth_dir = root_path + 'EstimatedDepth/LaSOT/'

        self.cocodepth_dir = root_path + 'EstimatedDepth/COCO_densedepth/'

        self.cdtb_dir = '/home/yan/Data2/vot-workspace/sequences/'
