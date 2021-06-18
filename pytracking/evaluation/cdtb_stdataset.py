import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


class CDTBD_ST_Dataset(BaseDataset):
    """
    CDTB, RGB dataset, Depth dataset, Colormap dataset, RGB+depth
    """
    def __init__(self, dtype='colormap'):
        super().__init__()
        self.base_path = self.env_settings.cdtb_st_path
        self.sequence_list = self._get_sequence_list()
        self.dtype = dtype

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        start_frame = 1

        if self.dtype == 'color':
            ext = 'jpg'
        elif self.dtype in ['R', 'G', 'B', 'RColormap', 'GColormap', 'BColormap']:
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

        if self.dtype in ['sigmoid', 'sigmoid_depth', 'colormap_depth', 'colormap', 'normalized_depth', 'raw_depth', 'centered_colormap', 'centered_normalized_depth', 'centered_raw_depth']:
            group = 'depth'
        elif self.dtype in ['color', 'R', 'G', 'B', 'RColormap', 'GColormap', 'BColormap']:
            group = 'color'
        else:
            group = self.dtype

        if self.dtype in ['rgbd', 'rgbcolormap']:
            frames = [{'color': '{base_path}/{sequence_path}/color/{frame:0{nz}}.jpg'.format(base_path=self.base_path,sequence_path=sequence_path, frame=frame_num, nz=nz),
                       'depth': '{base_path}/{sequence_path}/depth/{frame:0{nz}}.png'.format(base_path=self.base_path,sequence_path=sequence_path, frame=frame_num, nz=nz)
                       }for frame_num in range(start_frame, end_frame+1)]
            # color_frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.jpg'.format(base_path=self.base_path,
            #                 sequence_path=sequence_path, frame=frame_num, nz=nz)
            #                 for frame_num in range(start_frame, end_frame+1)]
            # frames = {'color': color_frames, 'depth': depth_frames}


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

        return Sequence(sequence_name, frames, 'cdtb', ground_truth_rect, dtype=self.dtype)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list= ['backpack_blue_1',
                        'backpack_robotarm_lab_occ_1',
                        'backpack_robotarm_lab_occ_3',
                        'backpack_room_noocc_1_1',
                        'bag_outside_1',
                        'bag_outside_2',
                        'bag_outside_3',
                        'bicycle2_outside_1',
                        'bicycle2_outside_2',
                        'bicycle_outside_1',
                        'bottle_box_1',
                        'bottle_box_2',
                        'bottle_box_3',
                        'bottle_box_4',
                        'bottle_box_5',
                        'bottle_room_noocc_1_1',
                        'bottle_room_occ_1_1',
                        'bottle_room_occ_1_2',
                        'box1_outside_1',
                        'box_darkroom_noocc_10_1',
                        'box_darkroom_noocc_1_1',
                        'box_darkroom_noocc_2_1',
                        'box_darkroom_noocc_3_1',
                        'box_darkroom_noocc_4_1',
                        'box_darkroom_noocc_5_1',
                        'box_darkroom_noocc_6_1',
                        'box_darkroom_noocc_7_1',
                        'box_darkroom_noocc_8_1',
                        'box_darkroom_noocc_9_1',
                        'boxes_backpack_room_occ_1_1',
                        'boxes_backpack_room_occ_1_2',
                        'boxes_backpack_room_occ_1_3',
                        'boxes_humans_room_occ_1_1',
                        'boxes_humans_room_occ_1_2',
                        'boxes_humans_room_occ_1_3',
                        'boxes_humans_room_occ_1_4',
                        'boxes_humans_room_occ_1_5',
                        'boxes_office_occ_1_1',
                        'boxes_office_occ_1_2',
                        'boxes_room_occ_1_1',
                        'boxes_room_occ_1_2',
                        'box_humans_room_occ_1_1',
                        'box_room_noocc_1_1',
                        'box_room_noocc_1_2',
                        'box_room_noocc_2_1',
                        'box_room_noocc_3_1',
                        'box_room_noocc_4_1',
                        'box_room_noocc_5_1',
                        'box_room_noocc_6_1',
                        'box_room_noocc_7_1',
                        'box_room_noocc_8_1',
                        'box_room_noocc_9_1',
                        'box_room_noocc_9_2',
                        'box_room_occ_1_1',
                        'box_room_occ_1_2',
                        'box_room_occ_1_3',
                        'box_room_occ_2_1',
                        'box_room_occ_2_2',
                        'box_room_occ_2_3',
                        'cartman_1',
                        'cartman_2',
                        'cartman_3',
                        'cartman_4',
                        'cartman_robotarm_lab_noocc_1',
                        'cart_room_occ_1_1',
                        'cart_room_occ_1_2',
                        'case_1',
                        'case_2',
                        'case_3',
                        'container_room_noocc_1_1',
                        'dog_outside_1',
                        'human_entry_occ_1_3',
                        'human_entry_occ_2_1',
                        'humans_corridor_occ_1_1',
                        'humans_corridor_occ_1_3',
                        'humans_corridor_occ_2_A_1',
                        'humans_corridor_occ_2_A_5',
                        'humans_corridor_occ_2_B_1',
                        'humans_corridor_occ_2_B_4',
                        'humans_longcorridor_staricase_occ_1_1',
                        'humans_longcorridor_staricase_occ_1_3',
                        'humans_shirts_room_occ_1_A_1',
                        'humans_shirts_room_occ_1_A_2',
                        'humans_shirts_room_occ_1_B_3',
                        'jug_1',
                        'jug_2',
                        'jug_3',
                        'mug_ankara_1',
                        'mug_ankara_2',
                        'mug_ankara_3',
                        'mug_gs_1',
                        'mug_gs_2',
                        'mug_gs_3',
                        'paperpunch_1',
                        'paperpunch_2',
                        'paperpunch_3',
                        'person_outside_1',
                        'person_outside_2',
                        'person_outside_3',
                        'robot_corridor_noocc_1_1',
                        'robot_corridor_occ_1_1',
                        'robot_corridor_occ_1_3',
                        'robot_human_corridor_noocc_1_B_1',
                        'robot_human_corridor_noocc_2_1',
                        'robot_human_corridor_noocc_3_A_1',
                        'robot_human_corridor_noocc_3_B_1',
                        'robot_lab_occ_1',
                        'robot_lab_occ_2',
                        'robot_lab_occ_3',
                        'teapot_1',
                        'teapot_2',
                        'tennis_ball_1',
                        'tennis_ball_2',
                        'thermos_office_noocc_1_1',
                        'thermos_office_occ_1_1',
                        'thermos_office_occ_1_2',
                        'thermos_office_occ_1_3',
                        'toy_office_noocc_1_1',
                        'toy_office_occ_1_1',
                        'toy_office_occ_1_2',
                        'toy_office_occ_1_3',
                        'trashcan_room_occ_1_1',
                        'trashcan_room_occ_1_2',
                        'trashcan_room_occ_1_3',
                        'trashcan_room_occ_1_4',
                        'trashcan_room_occ_1_6',
                        'trashcan_room_occ_1_7',
                        'trashcans_room_occ_1_A_1',
                        'trashcans_room_occ_1_A_2',
                        'trashcans_room_occ_1_B_1',
                        'trashcans_room_occ_1_B_2',
                        'trashcans_room_occ_1_B_3',
                        'trendNetBag_outside_1',
                        'trendNetBag_outside_2',
                        'trendNet_outside_1',
                        'trendNet_outside_2',
                        'trophy_outside_1',
                        'trophy_outside_2',
                        'trophy_outside_3',
                        'trophy_room_noocc_1_1',
                        'trophy_room_occ_1_1',
                        'trophy_room_occ_1_2',
                        'trophy_room_occ_1_3',
                        'two_mugs_1',
                        'two_mugs_2',
                        'two_mugs_3',
                        'two_mugs_4',
                        'two_mugs_5',
                        'two_mugs_6',
                        'two_tennis_balls_1',
                        'two_tennis_balls_2',
                        'two_tennis_balls_3',
                        'XMG_outside_1',
                        'XMG_outside_2']

        return sequence_list
