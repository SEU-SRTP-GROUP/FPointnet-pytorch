import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from collections import OrderedDict
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT

class Config(object):
    def __init__(self):
        '''
        说明： 配置类，保存着 FPointNet所需要的所有参数，如果需要参数就在Config类的构造函数里面加,参数用大写加下划线表示 eg: IS_TRAINING, BN_DECAY
        '''
        self.IS_TRAINING = True           # 是否训练 bn 层 在eval的时候应该为 false
        self.BN_DECAY = 0.9               # bn 层 的 momentum 参数
class FPointNet(nn.Module):
    def __init__(self,config):
        '''
        说明： 1 把需要训练的层在这里定义， 层的命名规则 层名_子模块缩写_编号 eg: self.conv_box3d_1
              2  end_points 保存着每一个模块的计算结果，是一个字典，在每一个模块结束之后对 end_points的内容进行简要说明
        '''
        super(FPointNet,self).__init__()
        self.config = config
        self.end_points = {}             #所有输出构成的字典
        self.conv_3dbox_1 = nn.Sequential(OrderedDict([
            ('conv_3dbox_1', nn.Conv2d(3,128,[1,1])),
            ('bn_3dbox_1', nn.BatchNorm2d(128,momentum=self.config.BN_DECAY,affine=self.config.IS_TRAINING))
            ('relu_3dbox_1', nn.ReLU()),
        ]))
        self.conv_3dbox_2 = nn.Sequential(OrderedDict([
            ('conv_3dbox_2', nn.Conv2d(128, 128, [1, 1])),
            ('bn_3dbox_2', nn.BatchNorm2d(128, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING))
            ('relu_3dbox_2', nn.ReLU()),
        ]))
        self.conv_3dbox_3 = nn.Sequential(OrderedDict([
            ('conv_3dbox_3', nn.Conv2d(128, 256, [1, 1])),
            ('bn_3dbox_3', nn.BatchNorm2d(256, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING))
            ('relu_3dbox_3', nn.ReLU()),
        ]))
        self.conv_3dbox_4 = nn.Sequential(OrderedDict([
            ('conv_3dbox_4', nn.Conv2d(256, 512, [1, 1])),
            ('bn_3dbox_4', nn.BatchNorm2d(512, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING))
            ('relu_3dbox_4', nn.ReLU()),
        ]))
        self.fc_3dbox_1 = nn.Sequential(OrderedDict([
            ('fc_3dbox_1', nn.Linear(515,512)),
            ('bn_3dbox_5', nn.BatchNorm2d(512, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING))
            ('relu_3dbox_5', nn.ReLU()),
        ]))
        self.fc_3dbox_2 = nn.Sequential(OrderedDict([
            ('fc_3dbox_2', nn.Linear(515, 256)),
            ('bn_3dbox_6', nn.BatchNorm2d(256, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING))
            ('relu_3dbox_6', nn.ReLU()),
        ]))
        self.fc_3dbox_3 = nn.Linear(256, 3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
        
    def get_3d_box_estimation_v1_net(self,object_point_cloud,one_hot_vec):
        '''
        3D box 回归模块
        @author :chonepieceyb
        :param object_point_cloud:  经过上一个模块处理后的点云数据  shape: (batch,C,Mask_num)  point clouds in object coordinate chanels should be 3
        :param one_hot_vec:  tensor in shape （batch,3) , length-3 vectors indicating predicted object type
        :return:  tensor in shape (B,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4) including box centers, heading bin class scores and residuals, and size cluster scores and residuals
        '''
        num_point = object_point_cloud.size()[2]
        net = torch.unsqueeze(object_point_cloud,3)         # change shape to (batch,C,M,1)
        net = self.conv_3dbox_1(net)                        # conv_block conv+bn+relu ,  [B,3,M,1]->[B,128,M,1]
        net = self.conv_3dbox_2(net)                        # conv_block conv+bn+relu ,  [B,128,M,1]->[B,128,M,1]
        net = self.conv_3dbox_3(net)                        # conv_block conv+bn+relu ,  [B,128,M,1]->[B,256,M,1]
        net = self.conv_3dbox_4(net)                        # conv_block conv+bn+relu ,  [B,256,M,1]->[B,512,M,1]
        net = F.max_pool2d(net,(num_point,1))               # max_pool layer   [B,512,M,1]->[B,512,1,1]
        net = net.view(-1,512)                     # change shape to [B,512]
        net = torch.cat((net,one_hot_vec),dim=1)            #  shape  [B,512+3]
        net = self.fc_3dbox_1(net)
        net = self.fc_3dbox_2(net)
        output = self.fc_3dbox_3(net)
        return output


    def forward(self, point_cloud, one_hot_vec):
        '''
        等同于原本的 get_model
        @author:chonepieceyb
        :param point_cloud:  tensor in shape (B,4,N) : 4 (x y z + intensity)
        :param one_hot_vec:  shape (B,3) predicted object type
        :return:   end_points: dict (map from name strings to TF tensors)
        '''
