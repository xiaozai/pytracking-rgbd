import sys
sys.path.append('/home/sgn/Data1/yan/pytracking-rgbd/')
sys.path.append('/home/sgn/Data1/yan/pytracking-rgbd/pytracking')
sys.path.append('/home/sgn/Data1/yan/pytracking-rgbd/ltr')

from ltr.admin.loading import torch_load_legacy
import torch

if __name__ == '__main__':


    ''' We train networks one the Machince with Torch 1.7.1, but we want to test on torch 1.4.0 '''

    net_path = '/home/sgn/Data1/yan/pytracking-models/checkpoints/ltr/bbreg/DeT_ATOM_Mean/ep0080.pth.tar'
    checkpoints = torch_load_legacy(net_path)
    # for key in checkpoints:
    #     print(key)
    checkpoints['constructor'].fun_name = 'atom_resnet18_DeT'
    torch.save(checkpoints, '/home/sgn/Data1/yan/pytracking-models/checkpoints/ltr/bbreg/DeT_ATOM_Mean/DeT_ATOM_Mean.pth.tar', _use_new_zipfile_serialization=False)
