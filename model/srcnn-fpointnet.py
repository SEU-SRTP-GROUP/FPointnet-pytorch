import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
FPOINTNET_DIR = os.path.join(ROOT_DIR,'fpointnet')
sys.path.append(os.path.join(FPOINTNET_DIR, 'model'))

import frustum_pointnets_v1 as FP

class SRCNNFPointNet(nn.Module):
    def __init__(self):
        super(SRCNNFPointNet, self).__init__()
        # fpointnet block
        self._fpointnet =  FP.FPointNet()

    def forward(self, point_cloud, one_hot_vec):

        return self._fpointnet.forward(point_cloud,one_hot_vec)


if __name__ == '__main__' :

    batch_size = 2
    N = 1024
    test_input = torch.rand((batch_size, 4, N))
    test_one_hot = torch.rand((batch_size, 3))
    srcnn_fpointnet = SRCNNFPointNet()

    # 随机生成数据
    batch_size = 2
    N = 1024
    test_input = torch.rand((batch_size, 4, N))
    test_one_hot = torch.rand((batch_size, 3))

    # 打印参数
    for name,parm in srcnn_fpointnet.named_parameters():
        print(name,"----------",parm)
        print('-------------------------------------')

    end_point = srcnn_fpointnet.forward(test_input,test_one_hot)

    for value in end_point.values():
        print(value)