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
from model_util import point_cloud_masking
from model_util import parse_output_to_tensors
class Config(object):
    def __init__(self):
        '''
        说明： 配置类，保存着 FPointNet所需要的所有参数，如果需要参数就在Config类的构造函数里面加,参数用大写加下划线表示 eg: IS_TRAINING, BN_DECAY
        '''
        self.INPUT_CHANNEL = 4            # 网络输入通道数
        self.OBJECT_INPUT_CHANNEL =3      # center regress 模块的输出通道数  3 = xyz_only
        self.IS_TRAINING = True           # 是否训练 bn 层 在eval的时候应该为 false
        self.BN_DECAY = 0.9               # bn 层 的 momentum 参数
class FPointNet(nn.Module):
    def __init__(self,config=Config()):

        super(FPointNet,self).__init__()
        self.config = config
        '''
        end_points: {
            mask_logists : tensor in shape (B,2,N)
            mask:   (B,N)   dim=1  1： 选中 0:不选中
            stage1_center : 每一个proposal的估计的中心坐标  shape (B,3)
            center_boxnet : box回归网络中，预测得到的局部坐标系坐标 shape (B,3)
            heading_scores: 分别属于哪一种heading的得分   shape(B,NUM_HEADING_BIN)
            heading_residuals_normalized: 正则化后的heading 残差项 : shape(B,NUM_HEADING_BIN)
            heading_residuals   : 未正则化项  shape(B,NUM_HEADING_BIN)
            size_scores      :  分别属于哪一种物体的大小（car...） 的得分    shape(B,NUM_SIZE_CLUSTER)
            size_residuals_normalized ： 正则化后的box size 残差项         shape(B,3,NUM_SIZE_CLUSTER)
            size_residuals:     未正则化项
            center            : 全局坐标             （B,3)
        }
        '''
        self.end_points = {}             #所有输出构成的字典

        '''
        #author: Qiao
        实例分割模块用到的层
        '''
        self.get_instance_seg_1 = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv_seg_1", torch.nn.Conv2d(self.config.INPUT_CHANNEL,
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

        ##############   T-Net 参数模块  ######################

        self.conv_Tnet_1 =nn.Sequential(OrderedDict([
            ('conv_Tnet_1', nn.Conv2d(self.config.OBJECT_INPUT_CHANNEL, 128, [1, 1])),
            ('bn_Tnet_1', nn.BatchNorm2d(128, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
            ('relu_Tnet_1', nn.ReLU()),
        ]))
        self.conv_Tnet_2 = nn.Sequential(OrderedDict([
            ('conv_Tnet_2', nn.Conv2d(self.config.OBJECT_INPUT_CHANNEL, 128, [1, 1])),
            ('bn_Tnet_2', nn.BatchNorm2d(128, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
            ('relu_Tnet_2', nn.ReLU()),
        ]))
        self.conv_Tnet_3 = nn.Sequential(OrderedDict([
            ('conv_Tnet_3', nn.Conv2d(self.config.OBJECT_INPUT_CHANNEL, 256, [1, 1])),
            ('bn_Tnet_3', nn.BatchNorm2d(128, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
            ('relu_Tnet_3', nn.ReLU()),
        ]))
        self.fc_Tnet_1 = nn.Sequential(OrderedDict([
            ('fc_Tnet_1', nn.Linear(256+3, 256)),
            ('bn_Tnet_4', nn.BatchNorm2d(512, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
            ('relu_Tnet_4', nn.ReLU()),
        ]))
        self.fc_Tnet_2 = nn.Sequential(OrderedDict([
            ('fc_Tnet_2', nn.Linear(256, 128)),
            ('bn_Tnet_5', nn.BatchNorm2d(128, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
            ('relu_Tnet_5', nn.ReLU()),
        ]))
        self.fc_Tnet_3 = nn.Linear(128,3)
        ##############   3d box 回归参数 ######################
        self.conv_3dbox_1 = nn.Sequential(OrderedDict([
            ('conv_3dbox_1', nn.Conv2d(3, 128, [1, 1])),
            ('bn_3dbox_1', nn.BatchNorm2d(128, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
            ('relu_3dbox_1', nn.ReLU()),
        ]))
        self.conv_3dbox_2 = nn.Sequential(OrderedDict([
            ('conv_3dbox_2', nn.Conv2d(128, 128, [1, 1])),
            ('bn_3dbox_2', nn.BatchNorm2d(128, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
            ('relu_3dbox_2', nn.ReLU()),
        ]))
        self.conv_3dbox_3 = nn.Sequential(OrderedDict([
            ('conv_3dbox_3', nn.Conv2d(128, 256, [1, 1])),
            ('bn_3dbox_3', nn.BatchNorm2d(256, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
            ('relu_3dbox_3', nn.ReLU()),
        ]))
        self.conv_3dbox_4 = nn.Sequential(OrderedDict([
            ('conv_3dbox_4', nn.Conv2d(256, 512, [1, 1])),
            ('bn_3dbox_4', nn.BatchNorm2d(512, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
            ('relu_3dbox_4', nn.ReLU()),
        ]))
        self.fc_3dbox_1 = nn.Sequential(OrderedDict([
            ('fc_3dbox_1', nn.Linear(515, 512)),
            ('bn_3dbox_5', nn.BatchNorm2d(512, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
            ('relu_3dbox_5', nn.ReLU()),
        ]))
        self.fc_3dbox_2 = nn.Sequential(OrderedDict([
            ('fc_3dbox_2', nn.Linear(515, 256)),
            ('bn_3dbox_6', nn.BatchNorm2d(256, momentum=self.config.BN_DECAY, affine=self.config.IS_TRAINING)),
            ('relu_3dbox_6', nn.ReLU()),
        ]))
        self.fc_3dbox_3 = nn.Linear(256, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4)

    def get_instance_seg_v1_net(self, point_cloud, one_hot_vec):
        '''
        @author： Qiao
        实例分割网络
        notice：tensorflow是 NHWC  pytorch为 NCHW 需要调整

        Input:
            point_cloud: TF tensor in shape (B,4,N)
                frustum point clouds with XYZ and intensity in point channels
                XYZs are in frustum coordinate
            one_hot_vec: TF tensor in shape (B,3)
                length-3 vectors indicating predicted object type
            end_points: dict
        Output:
            logits: TF tensor in shape (B,2,N), scores for bkg/clutter and object
            end_points: dict
        '''
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[2].value

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

        logits = torch.squeeze(logits, 2) # BxCxN
        return logits

    def get_3d_box_estimation_v1_net(self,object_point_cloud,one_hot_vec):
        '''
        3D box 回归模块
        @author :chonepieceyb
        :param object_point_cloud:  经过上一个模块处理后的点云数据  shape: (batch,Mask_num,C)  point clouds in object coordinate
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

    def get_center_regression_net(self,object_point_cloud,one_hot_vec):
        '''
        @author chonepieceyb  T-Net 原本放在 model_utils里面，因为里面有参数需要训练 所以放在了类里
        :param object_point_cloud:  tensor in shape (B,C,num_points) point clouds in 3D mask coordinate
        :param one_hot_vec:   TF tensor in shape (B,3)  length-3 vectors indicating predicted object type
        :return:  predicted_center:  tensor in shape (B,3)
        '''
        num_point = object_point_cloud.size()[2]
        net = torch.unsqueeze(object_point_cloud,3)       #change shape to ( B,C,num_points,1)
        net = self.conv_Tnet_1(net)                        # conv_block conv+bn+relu ,  [B,C,M,1]->[B,128,M,1]
        net = self.conv_Tnet_2(net)                         # conv_block conv+bn+relu ,  [B,128,M,1]->[B,128,M,1]
        net = self.conv_Tnet_3(net)                          # conv_block conv+bn+relu ,  [B,128,M,1]->[B,256,M,1]
        net = F.max_pool2d(net,(num_point,1))              # max_pool layer   [B,256,M,1]->[B,256,1,1]
        net = net.view(-1,256)                           # (B,256)
        net = torch.cat((net,one_hot_vec))                # (B,259)
        net = self.fc_Tnet_1(net)                        # fc+bn+relu    [B,259]->[B,256]
        net = self.fc_Tnet_2(net)                         # fc+bn+relu    [B,256]->[B,128]
        predicted_center = self.fc_Tnet_3(net)            # fc           [B,128]->[B,3]
        return predicted_center

    def forward(self, point_cloud,one_hot_vec):
        '''
        @ author chonepieceyb
        :param point_cloud:  tensor in shape (B,4,N)
                             frustum point clouds with XYZ and intensity in point channels
                             XYZs are in frustum coordinate
        :param one_hot_vec:  F tensor in shape (B,3)
            length-3 vectors indicating predicted object type
        :return:  end_points: dict (map from name strings to  tensors)
        '''
        logits =  self.get_instance_seg_v1_net(
            point_cloud, one_hot_vec
        )  # 这里接口可能有点问题
        self.end_points['mask_logists'] = logits
        # masking
        # select masked points and translate to masked points' centroid
        object_point_cloud_xyz, mask_xyz_mean ,self.end_points = \
            point_cloud_masking(point_cloud,logits,self.end_points)
        # T-Net and coordinate translation
        center_delta = self.get_center_regression_net(object_point_cloud_xyz,one_hot_vec)                             # 局部坐标系的回归, center_delta shape (B,3)
        stage1_center = center_delta+ mask_xyz_mean
        self.end_points['stage1_center'] = stage1_center
        object_point_cloud_xyz_new = object_point_cloud_xyz - torch.unsqueeze(center_delta,dim=2)                    # -(B,3,1)

        # Amodel Box Estimation PointNet
        output = self.get_3d_box_estimation_v1_net(object_point_cloud_xyz,one_hot_vec)

        # parse output to 3D box parameters
        self.end_points = parse_output_to_tensors (output,self.end_points)
        self.end_points['center'] =  self.end_points['center_boxnet'] + stage1_center # Bx3
        return self.end_points


if __name__ =='__main__':
    batch_size = 2
    N = 1024
    test_input = torch.rand((batch_size, 4, N))
    test_one_hot = torch.rand((batch_size, 3))
    fpointnet = FPointNet()
    print(fpointnet)
    output = fpointnet.forward(test_input, test_one_hot)