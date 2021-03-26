import sys
sys.path.append('../../pytracking-rgbd/')
sys.path.append('../../pytracking-rgbd/pytracking')
sys.path.append('../../pytracking-rgbd/ltr')

from ltr.admin.loading import torch_load_legacy
import torch

if __name__ == '__main__':


    ''' We train networks one the Machince with Torch 1.7.1, but we want to test on torch 1.4.0 '''

    net_path = '/home/yan/Data2/DeT-models/DeT_DiMP50_Max.pth.tar'
    checkpoints = torch_load_legacy(net_path)
    checkpoints['epoch'] = 50

    torch.save(checkpoints, net_path, _use_new_zipfile_serialization=False)
