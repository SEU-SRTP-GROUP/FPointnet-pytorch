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

        '''
        #author: Qiao
        实例分割模块用到的层
        '''
        self.get_instance_seg_1 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv_seg_1", torch.nn.Conv2d(self.config.point_cloud.get_shape()[1].value,
                                              64, 1, stride=1, padding=0)),
                    ("bn_seg_1", torch.nn.BatchNorm2d(64,
                                                      momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
                    ("relu_seg_1",torch.nn.ReLU()),
                ]))
        self.get_instance_seg_2 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv_seg_2", torch.nn.Conv2d(64,
                                              64, 1, stride=1, padding=0)),
                    ("bn_seg_2", torch.nn.BatchNorm2d(64,
                                                      momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
                    ("relu_seg_2",torch.nn.ReLU()),
                ]))
        # 这一层就结束得到64维局部特征point_feature
        self.get_instance_seg_3 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv_seg_3", torch.nn.Conv2d(64,
                                              64, 1, stride=1, padding=0)),
                    ("bn_seg_3", torch.nn.BatchNorm2d(64,
                                                      momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
                    ("relu_seg_3",torch.nn.ReLU()),
                ]))
        # 这一层就结束得到1024维全局特征global_feature
        self.get_instance_seg_4 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv_seg_4", torch.nn.Conv2d(64,
                                              128, 1, stride=1, padding=0)),
                    ("bn_seg_4", torch.nn.BatchNorm2d(128,
                                                      momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
                    ("relu_seg_4",torch.nn.ReLU()),
                ]))
        self.get_instance_seg_5 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv_seg_5", torch.nn.Conv2d(128,
                                              1024, 1, stride=1, padding=0)),
                    ("bn_seg_5", torch.nn.BatchNorm2d(1024,
                                                      momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
                    ("relu_seg_5",torch.nn.ReLU()),
                ]))
        self.get_instance_seg_pool_1 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("pool_seg_1", torch.nn.MaxPool2d(2))
                ]))
        # 然后需要拼接两个特征变成一个1088维的

        self.get_instance_seg_6 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv_seg_6", torch.nn.Conv2d(1088,
                                              512, 1, stride=1, padding=0)),
                    ("bn_seg_6", torch.nn.BatchNorm2d(512,
                                                      momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
                    ("relu_seg_6",torch.nn.ReLU()),
                ]))
        self.get_instance_seg_7 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv_seg_7", torch.nn.Conv2d(512,
                                              256, 1, stride=1, padding=0)),
                    ("bn_seg_7", torch.nn.BatchNorm2d(256,
                                                      momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
                    ("relu_seg_7",torch.nn.ReLU()),
                ]))
        self.get_instance_seg_8 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv_seg_8", torch.nn.Conv2d(256,
                                              128, 1, stride=1, padding=0)),
                    ("bn_seg_8", torch.nn.BatchNorm2d(128,
                                                      momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
                    ("relu_seg_8",torch.nn.ReLU()),
                ]))
        self.get_instance_seg_9 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv_seg_9", torch.nn.Conv2d(128,
                                              128, 1, stride=1, padding=0)),
                    ("bn_seg_9", torch.nn.BatchNorm2d(128,
                                                      momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
                    ("relu_seg_9",torch.nn.ReLU()),
                ]))
        self.get_instance_seg_dp_1 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("dp_seg_1", torch.nn.Dropout(p=0.5))
                ]))
        self.get_instance_seg_10 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv_seg_10", torch.nn.Conv2d(128,
                                              2, 1, stride=1, padding=0)),
                    # ("bn_seg_10", torch.nn.BatchNorm2d(2,
                    #                                   momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
                ]))

    def get_3d_box_estimation_v1_net(self,object_point_cloud,one_hot_vec):
        '''
        3D box 回归模块
        @author :chonepieceyb
        :param object_point_cloud:  经过上一个模块处理后的点云数据  shape: (batch,Mask_num,C)  point clouds in object coordinate
        :param object_point_cloud:  经过上一个模块处理后的点云数据  shape: (batch,C,Mask_num)  point clouds in object coordinate chanels should be 3
        :param one_hot_vec:  tensor in shape （batch,3) , length-3 vectors indicating predicted object type
        :return:  tensor in shape (B,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4) including box centers, heading bin class scores and residuals, and size cluster scores and residuals
        '''
        num_point = object_point_cloud.size()[1]
        net = torch.unsqueeze(object_point_cloud,2)         # change shape to (batch,M,1,C)
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

    def get_instance_seg_v1_net(self, point_cloud, one_hot_vec):
        '''
        @author： Qiao
        实例分割网络
        notice：tensorflow是 NHWC  pytorch为 NCHW 需要调整

        Input:
            point_cloud: TF tensor in shape (B,N,4)
                frustum point clouds with XYZ and intensity in point channels
                XYZs are in frustum coordinate
            one_hot_vec: TF tensor in shape (B,3)
                length-3 vectors indicating predicted object type
            end_points: dict
        Output:
            logits: TF tensor in shape (B,N,2), scores for bkg/clutter and object
            end_points: dict
        '''
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value

        # net = tf.expand_dims(point_cloud, 2)
        net = torch.unsqueeze(point_cloud, 3)

        net = self.get_instance_seg_1(net)
        net = self.get_instance_seg_2(net)
        point_feat = self.get_instance_seg_3(net)
        net = self.get_instance_seg_4(point_feat)
        net = self.get_instance_seg_5(net)
        global_feat = self.get_instance_seg_pool_1(net)

        # 把通道数拼起来 pytorch中为第二个
        global_feat = torch.cat([global_feat, torch.unsqueeze(torch.unsqueeze(one_hot_vec, 1), 1)], 1)

        global_feat_expand = torch.repeat(global_feat, [1, num_point, 1, 1])

        concat_feat = torch.cat([point_feat, global_feat_expand],3)

        net = self.get_instance_seg_6(concat_feat)
        net = self.get_instance_seg_7(net)
        net = self.get_instance_seg_8(net)
        net = self.get_instance_seg_9(net)
        net = self.get_instance_seg_dp_1(net)
        logits = self.get_instance_seg_10(net)

        logits = torch.squeeze(logits, 2) # BxNxC
        return logits, self.end_points
