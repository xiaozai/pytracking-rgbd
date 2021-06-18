import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os

class DepthTrack_ST_Dataset(BaseDataset):
    """
    CDTB, RGB dataset, Depth dataset, Colormap dataset, RGB+depth
    """
    def __init__(self, dtype='colormap'):
        super().__init__()
        self.base_path = self.env_settings.depthtrack_st_path
        self.sequence_list = self._get_sequence_list()
        self.dtype = dtype

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        start_frame = 1

        if self.dtype in ['color', 'R', 'G', 'B', 'RColormnap', 'GColormap', 'BColormap']:
            ext = 'jpg'
        elif self.dtype == 'rgbd':
            ext = ['jpg', 'png'] # Song not implemented yet
        else:
            ext = 'png'

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        if self.dtype in ['colormap', 'normalized_depth', 'raw_depth', 'centered_colormap', 'centered_normalized_depth', 'centered_raw_depth']:
            group = 'depth'
        elif self.dtype in ['color', 'R', 'G', 'B', 'RColormnap', 'GColormap', 'BColormap']:
            group = 'color'
        else:
            group = self.dtype

        if self.dtype in ['rgbd', 'rgbcolormap']:
            depth_frames = ['{base_path}/{sequence_path}/depth/{frame:0{nz}}.png'.format(base_path=self.base_path,
                            sequence_path=sequence_path, frame=frame_num, nz=nz)
                            for frame_num in range(start_frame, end_frame+1)]
            color_frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.jpg'.format(base_path=self.base_path,
                            sequence_path=sequence_path, frame=frame_num, nz=nz)
                            for frame_num in range(start_frame, end_frame+1)]
            # frames = {'color': color_frames, 'depth': depth_frames}
            frames = []
            for c_path, d_path in zip(color_frames, depth_frames):
                frames.append({'color': c_path, 'depth': d_path})

        else:
            frames = ['{base_path}/{sequence_path}/{group}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                      sequence_path=sequence_path, group=group, frame=frame_num, nz=nz, ext=ext)
                      for frame_num in range(start_frame, end_frame+1)]

        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)

        return Sequence(sequence_name, frames, 'depthtrack', ground_truth_rect, dtype=self.dtype)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        # sequence_list = os.listdir(self.base_path)
        # sequence_list = [seq for seq in sequence_list if os.path.isfile(os.path.join(self.base_path, seq, 'groundtruth.txt'))]
        sequence_list= ['adapter01_indoor_1',
                        'adapter01_indoor_2',
                        'adapter01_indoor_3',
                        'adapter01_indoor_4',
                        'backpack_indoor_1',
                        'backpack_indoor_2',
                        'backpack_indoor_3',
                        'bag01_indoor_1',
                        'bag01_indoor_2',
                        'bag01_indoor_3',
                        'bag01_indoor_4',
                        'bag02_indoor_1',
                        'bag02_indoor_2',
                        'bag02_indoor_3',
                        'ball01_wild_1',
                        'ball01_wild_2',
                        'ball06_indoor_1',
                        'ball06_indoor_2',
                        'ball10_wild_1',
                        'ball10_wild_2',
                        'ball10_wild_3',
                        'ball10_wild_4',
                        'ball11_wild_1',
                        'ball11_wild_2',
                        'ball11_wild_3',
                        'ball11_wild_4',
                        'ball15_wild_1',
                        'ball18_indoor_1',
                        'ball20_indoor_1',
                        'ball20_indoor_2',
                        'ball20_indoor_3',
                        'ball20_indoor_4',
                        'bandlight_indoor_1',
                        'beautifullight02_indoor_1',
                        'book03_indoor_1',
                        'bottle04_indoor_1',
                        'bottle04_indoor_2',
                        'bottle04_indoor_3',
                        'bottle04_indoor_4',
                        'card_indoor_1',
                        'cat01_indoor_1',
                        'cat01_indoor_2',
                        'cat01_indoor_3',
                        'colacan03_indoor_1',
                        'colacan03_indoor_2',
                        'colacan03_indoor_3',
                        'colacan03_indoor_4',
                        'colacan03_indoor_5',
                        'colacan03_indoor_6',
                        'cube02_indoor_1',
                        'cube02_indoor_2',
                        'cube02_indoor_3',
                        'cube02_indoor_4',
                        'cube02_indoor_5',
                        'cube02_indoor_6',
                        'cube03_indoor_1',
                        'cube05_indoor_1',
                        'cube05_indoor_2',
                        'cube05_indoor_3',
                        'cube05_indoor_4',
                        'cube05_indoor_5',
                        'cube05_indoor_6',
                        'cube05_indoor_7',
                        'cup01_indoor_1',
                        'cup01_indoor_2',
                        'cup02_indoor_1',
                        'cup04_indoor_1',
                        'cup04_indoor_2',
                        'cup04_indoor_3',
                        'cup04_indoor_4',
                        'cup12_indoor_1',
                        'cup12_indoor_2',
                        'developmentboard_indoor_1',
                        'developmentboard_indoor_2',
                        'developmentboard_indoor_3',
                        'developmentboard_indoor_4',
                        'duck03_wild_1',
                        'duck03_wild_2',
                        'dumbbells01_indoor_1',
                        'dumbbells01_indoor_2',
                        'earphone01_indoor_1',
                        'file01_indoor_1',
                        'file01_indoor_2',
                        'file01_indoor_3',
                        'flag_indoor_1',
                        'glass01_indoor_1',
                        'glass01_indoor_2',
                        'glass01_indoor_3',
                        'hand01_indoor_1',
                        'hand01_indoor_2',
                        'human02_indoor_1',
                        'human02_indoor_2',
                        'lock_wild_1',
                        'mobilephone03_indoor_1',
                        'mobilephone03_indoor_2',
                        'notebook01_indoor_1',
                        'notebook01_indoor_2',
                        'notebook01_indoor_3',
                        'notebook01_indoor_4',
                        'notebook01_indoor_5',
                        # 'notebook01_indoor_6',
                        'notebook01_indoor_7',
                        'pigeon01_wild_1',
                        'pigeon02_wild_1',
                        'pigeon04_wild_1',
                        'pot_indoor_1',
                        'pot_indoor_2',
                        'pot_indoor_3',
                        'pot_indoor_4',
                        'roller_indoor_1',
                        'roller_indoor_2',
                        'roller_indoor_3',
                        'shoes02_indoor_1',
                        'shoes02_indoor_2',
                        'squirrel_wild_1',
                        'squirrel_wild_2',
                        'squirrel_wild_3',
                        'stick_indoor_1',
                        'toiletpaper01_indoor_1',
                        'toiletpaper01_indoor_2',
                        'toiletpaper01_indoor_3',
                        'toy02_indoor_1',
                        'toy09_indoor_1',
                        'ukulele01_indoor_1',
                        'ukulele01_indoor_2',
                        'yogurt_indoor_1']

        return sequence_list
