import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


class CDTBDColormapValDataset(BaseDataset):
    """
    CDTB, RGB dataset
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.cdtb_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        ext = 'jpg'
        start_frame = 1

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        frames = ['{base_path}/{sequence_path}/colormap/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                  sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
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
        return Sequence(sequence_name, frames, 'cdtb_dcolormap', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list= ['backpack_room_noocc_1',
                        'bicycle_outside',
                        'box1_outside',
                        'box_darkroom_noocc_4',
                        'box_darkroom_noocc_8',
                        'box_room_noocc_2',
                        'box_room_noocc_9',
                        'boxes_humans_room_occ_1',
                        'cartman',
                        'dog_outside',
                        'humans_corridor_occ_2_A',
                        'humans_shirts_room_occ_1_B',
                        'jug',
                        'paperpunch',
                        'robot_human_corridor_noocc_1_A',
                        'robot_human_corridor_noocc_3_B',
                        'tennis_ball',
                        'thermos_office_noocc_1',
                        'trashcan_room_occ_1',
                        'XMG_outside',
                        ]

        return sequence_list
