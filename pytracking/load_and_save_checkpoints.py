import sys
sys.path.append('/home/sgn/Data1/yan/pytracking-rgbd/')
sys.path.append('/home/sgn/Data1/yan/pytracking-rgbd/pytracking')
sys.path.append('/home/sgn/Data1/yan/pytracking-rgbd/ltr')

from ltr.admin.loading import torch_load_legacy

if __name__ == '__main__':


    ''' We train networks one the Machince with Torch 1.7.1, but we want to test on torch 1.4.0 '''
    
    net_path = '/home/sgn/Data1/yan/pytracking-models/checkpoints/ltr/dimp/DeT_DiMP50_DO/ep0050.pth.tar'
    checkpoints = torch_load_legacy(net_path)
    # for key in checkpoints:
    #     print(key)
    torch.save(checkpoints, '/home/sgn/Data1/yan/pytracking-models/checkpoints/ltr/dimp/DeT_DiMP50_DO/DeT_DiMP50_DO.pth.tar', _use_new_zipfile_serialization=False)
