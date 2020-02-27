import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):
    def __init__(self):
        '''
        说明： 配置类，保存着 FPointNet所需要的所有参数，如果需要参数就在Config类的构造函数里面加,参数用大写加下划线表示 eg: IS_TRAINING, BN_DECAY
        '''

class FPointNet(nn.Module):
    def __init__(self,config):
        '''
        说明： 1 把需要训练的层在这里定义， 层的命名规则 层名_子模块缩写_编号 eg: self.conv_box3d_1
              2  end_points 保存着每一个模块的计算结果，是一个字典，在每一个模块结束之后对 end_points的内容进行简要说明
        '''
        super(FPointNet,self).__init__()
        self.config = config
        self.end_points = {}             #所有输出构成的字典

    def get_3d_box_estimation_v1_net(self,object_point_cloud,one_hot_vec):
        '''
        3D box 回归模块
        @author :chonepieceyb
        :param object_point_cloud:  经过上一个模块处理后的点云数据  shape: (batch,Mask_num,C)  point clouds in object coordinate
        :param one_hot_vec:  tensor in shape （batch,3) , length-3 vectors indicating predicted object type
        :return:  tensor in shape (B,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4) including box centers, heading bin class scores and residuals, and size cluster scores and residuals
        '''
        num_point = object_point_cloud.size()[1]
        net = torch.unsqueeze(object_point_cloud,2)         # change shape to (batch,M,1,C)
