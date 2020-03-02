import numpy as np
import torch
import os
import sys
import torch
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8 # one cluster for each type 每种类型一个集群
NUM_OBJECT_POINT = 512
g_type2class={'Car':0, 'Van':1, 'Truck':2, 'Pedestrian':3,
              'Person_sitting':4, 'Cyclist':5, 'Tram':6, 'Misc':7}
g_class2type = {g_type2class[t]:t for t in g_type2class}#与上面相反，数字对应字符串
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
#np.array,存储单一数据类型的多维数组,避免浪费内存和CPU。
g_type_mean_size = {'Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                    'Van': np.array([5.06763659,1.9007158,2.20532825]),
                    'Truck': np.array([10.13586957,2.58549199,3.2520595]),
                    'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
                    'Person_sitting': np.array([0.80057803,0.5983815,1.27450867]),
                    'Cyclist': np.array([1.76282397,0.59706367,1.73698127]),
                    'Tram': np.array([16.17150617,2.53246914,3.53079012]),
                    'Misc': np.array([3.64300781,1.54298177,1.92320313])}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) # size clustrs
#array([[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.]])
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i,:] = g_type_mean_size[g_class2type[i]]
    #copy

def gather_object_pc(point_cloud,mask,npoints=512):
    '''
    根据mask 和 npoints 取出分割得到的点 如果点的数目 > 512 就随机采样
    :param point_cloud:   shape:(B,C,N)
    :param mask:           shape(B,N)  0 picked 1 pick
    :param npoints:       max num of points to keep
    :return:        object_pc:  tensor in shape (B,C,npoint)
                    indices:  int tensor in shape [B,npoint]  important!! 因为 pytorch 和 tensorflow 在
                     gather的 api存在一些差距 n, npoint 表示对每一个batch数据的 index ，采用这种方法 应该 配合 index_select使用 遍历batch
    '''
    #根据mask 计算 indices
    size = mask.size()
    indices = torch.zeros((size[0],npoints))
    object_pc = torch.tensor((size[0],point_cloud.size()[1],npoints))   # shape (B,C,npoints)
    for b in range(size[0]):
        pos_indices = torch.nonzero(mask[b,:])      # shape of b_i is [N2]
        length = len(pos_indices)
        if length >0:
            if length > npoints:
                #采样
                choice_index = torch.randperm(length)[0:npoints]
            else:
                choice_index = torch.randint(0,length,[npoints-length])       #随机重复采样 填补空缺    shape [N3]
                choice_index = torch.cat((torch.arange(0,length),choice_index),dim=0)
        indices[b] =  pos_indices [choice_index]
        object_pc[b] = torch.index_select(point_cloud[b],dim=1,index=indices[b])
    return object_pc, indices


def point_cloud_masking(point_cloud, logits, end_points, xyz_only=True):
    '''
    @author chonepieceyb
    :param point_cloud:   input  tensor in shape (B,C,N)
    :param logits:        tensor in shape (B,2,N)
    :param end_points:    dict of FPoint cal result
    :param xyz_only:      if True only return XYZ channels
    :return:        object_point_cloud:  tensor in shape (B,M,3)
                    for simplicity we only keep XYZ here
                    M = NUM_OBJECT_POINT as a hyper-parameter
                    mask_xyz_mean:  tensor in shape (B,3)     the mean value of all points' xyz
    '''
    batch_size = point_cloud.size()[0]
    num_point = point_cloud.size()[2]
    mask = torch.index_select(point_cloud, dim=1, index=torch.tensor([0])).le(
        torch.index_select(point_cloud, dim=1, index=torch.tensor([0])))  # dim=1 index=0 和 index =1做比较 得到mask  B,1,N
    mask = mask.float()  # 转化为 float 方便计算 , (B,1,N)
    point_cloud_xyz = torch.index_select(point_cloud, dim=1,
                                         index=torch.tensor([0, 1, 2]))  # 只选择 通道的 xyz , shape [B,3,N]
    mask_xyz_mean = torch.mean(
        point_cloud_xyz * mask.repeat(1, 3, 1), dim=2, keepdim=True
    )  # shape (B,3,1)
    mask = torch.squeeze(mask, dim=1)  # change shape to (B,N)
    end_points["mask"] = mask         # mask
    point_cloud_xyz_stage1 = point_cloud_xyz - mask_xyz_mean.repeat(1,1,num_point)    # 转换到局部坐标系   (B,3,N)

    if xyz_only:
        point_cloud_stage1 = point_cloud_xyz_stage1            # 如果只包含 xyz坐标
    else:
        point_cloud_features = point_cloud[:,3:,:]            # 获取额外的 feature     (B, C-3,N )
        point_cloud_stage1 = torch.cat((point_cloud_xyz_stage1,point_cloud_features),dim=1)   #   (B,C,N)
    num_channels = point_cloud_stage1.size()[1]
    object_point_cloud = gather_object_pc(point_cloud_stage1,mask,NUM_OBJECT_POINT)
    object_point_cloud.view((batch_size,num_channels,NUM_OBJECT_POINT))
    return object_point_cloud, torch.squeeze(mask_xyz_mean,dim=2),end_points