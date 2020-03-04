import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3)) #   size clustrs
#array([[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.],[ 0.,  0.,  0.]])
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i,:] = g_type_mean_size[g_class2type[i]]
    #copy

def init_fpointnet(net,use_xavier =True):
    '''
    初始化 net的权重的函数 ,
    @author: chonepieceyb
    :param net:         要初始化的网络
    :param use_xavier:  使用 xavier_uniform 初始化方式。 如果为ture 就采用pytorch 默认的初始化方式（应该是 uniform?)
    '''
    if use_xavier:
        for m in net.modules():
            if isinstance(m,nn.Conv2d):                   # 初始化 2D 卷积层
                nn.init.xavier_uniform_(m.weight)
                if m.bias !=None:
                    nn.init.constant(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):              # bn层 两个参数 初始化为 1 和 0
                nn.init.constant(m.weight,1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m,nn.Linear):                 #初始化全连接层
                nn.init.xavier_uniform_(m.weight)
                if m.bias !=None:
                    nn.init.constant(m.bias, 0)

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
    object_pc = torch.zeros((size[0],point_cloud.size()[1],npoints))   # shape (B,C,npoints)
    for b in range(size[0]):
        pos_indices = torch.nonzero(mask[b,:])
        pos_indices = torch.squeeze(pos_indices,-1) # shape of b_i is [N2]
        length = len(pos_indices)
        if length >0:
            if length > npoints:
                #采样
                choice_index = torch.randperm(length)[0:npoints]
            else:
                choice_index = torch.randint(0,length,[npoints-length])       #随机重复采样 填补空缺    shape [N3]
                choice_index = torch.cat((torch.arange(0,length),choice_index),dim=0)
        indices[b] =  pos_indices [choice_index]
        object_pc[b] = torch.index_select(point_cloud[b],dim=1, index=indices[b].type(torch.LongTensor) )
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
    mask = torch.index_select(logits, dim=1, index=torch.tensor([0])).le(
        torch.index_select(logits, dim=1, index=torch.tensor([1])))  # dim=1 index=0 和 index =1做比较 得到mask  B,1,N
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
    object_point_cloud ,_= gather_object_pc(point_cloud_stage1,mask,NUM_OBJECT_POINT)
    object_point_cloud.view((batch_size,num_channels,NUM_OBJECT_POINT))
    return object_point_cloud, torch.squeeze(mask_xyz_mean,dim=2),end_points

def parse_output_to_tensors(output, end_points):
    '''
    @author: chonepieceyb
    :param output:  tensor in shape (B,3+2*NUM_HEADING_BIN+4*NUM_SIZE_CLUSTER)
    :param end_points: dict
    :return: end_points: dict (updated)
    '''
    batch_size = output.size()[0]
    center = torch.index_select(output,dim=1,index=torch.arange(0,3))
    end_points['center_boxnet']  = center

    heading_scores = torch.index_select(output,dim=1,index=torch.arange(3,3+NUM_HEADING_BIN))
    heading_residuals_normalized = torch.index_select(output,dim=1,index=torch.arange(3+NUM_HEADING_BIN, 3+2*NUM_HEADING_BIN))
    end_points['heading_scores'] = heading_scores # BxNUM_HEADING_BIN
    end_points['heading_residuals_normalized'] = \
        heading_residuals_normalized # BxNUM_HEADING_BIN (-1 to 1)
    end_points['heading_residuals'] = \
        heading_residuals_normalized * (np.pi / NUM_HEADING_BIN)  # BxNUM_HEADING_BIN
    size_scores = torch.index_select(output,dim=1,index=torch.arange( 3+2*NUM_HEADING_BIN, 3+2*NUM_HEADING_BIN+NUM_SIZE_CLUSTER))
    size_residuals_normalized =  torch.index_select(output,dim=1,index=torch.arange( 3+2*NUM_HEADING_BIN+NUM_SIZE_CLUSTER, 3+2*NUM_HEADING_BIN+4*NUM_SIZE_CLUSTER))
    size_residuals_normalized = size_residuals_normalized.view((batch_size,3,NUM_SIZE_CLUSTER))         # 这里采用 pytorch的深度 ， 3 表示 dh, dw ,dl ?
    end_points['size_scores'] = size_scores
    end_points['size_residuals_normalized'] = size_residuals_normalized
    g_mean_size_arr_tensor =  torch.tensor(g_mean_size_arr,dtype=torch.float32).permute(1,0)        # 交换维度，原本的是 tensorflow风格 深度在最后一维  (NUM_SIZE_CLUSTER, 3)-> (3,NUM_SIZE_CLUSTER)
    end_points['size_residuals'] = size_residuals_normalized * \
        torch.unsqueeze(g_mean_size_arr_tensor,dim=0)
    return end_points

def get_box3d_corners_helper(centers, headings, sizes):
    """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    # print '-----', centers
    N = centers.get_shape()[0].value
    l = sizes[:, 0].view(N, 1)  # (N,1)
    w = sizes[:, 1].view(N, 1)  # (N,1)
    h = sizes[:, 2].view(N, 1)  # (N,1)
    # print l,w,h    (N,1):三个应该均为[[a1][a2][a3]...]

    x_corners = torch.cat([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=1)  # (N,8)
    y_corners = torch.cat([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], dim=1)  # (N,8)
    z_corners = torch.cat([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=1)  # (N,8)

    # 以下操作为给x_corners三项增加一个维度再合并 所以(N,8)变为(N,3,8)
    corners = torch.cat([x_corners.unsqueeze(1), y_corners.unsqueeze(1), z_corners.unsqueeze(1)], dim=1)  # (N,3,8)
    # print x_corners, y_corners, z_corners
    c = torch.cos(headings).cuda()
    s = torch.sin(headings).cuda()
    ones = torch.ones([N], dtype=torch.float32).cuda()
    zeros = torch.zeros([N], dtype=torch.float32).cuda()

    # stack()矩阵拼接的函数，在axis维上拼接 与concat不同
    # tf.concat拼接的是除了拼接维度axis外其他维度的shape完全相同的张量，并且产生的张量的阶数不会发生变化
    # 而tf.stack则会在新的张量阶上拼接，产生的张量的阶数将会增加
    row1 = torch.stack([c, zeros, s], dim=1)  # (N,3)
    row2 = torch.stack([zeros, ones, zeros], dim=1)
    row3 = torch.stack([-s, zeros, c], dim=1)
    R = torch.cat([row1.unsqueeze(1), row2.unsqueeze(1), row3.unsqueeze(1)], dim=1)  # (N,3)->(N,1,3)->(N,3,3)
    corners_3d = torch.bmm(R, corners)  # (N,3,8)
    # tile(input,multiples) 用于在同一维度上的复制 multiples[1,1,8]  三个维度分别变为1,1,8倍
    corners_3d += centers.unsqueeze(2).repeat(1, 1, 8)  # (N,3)->(N,3,1)->(N,3,8)
    corners_3d = torch.transpose(corners_3d, 1, 2)  # (N,8,3)
    return corners_3d


def get_box3d_corners(center, heading_residuals, size_residuals):
    """
    Inputs:
        center: (B,3)
        heading_residuals: (B,NH)
        size_residuals: (B,3,NS)
    Outputs:
        box3d_corners: (B,NH,NS,8,3) tensor                NH NS 为开头定义的常量NUM_HEADING_BIN = 12 NUM_SIZE_CLUSTER = 8缩写
    """
    batch_size = center.get_shape()[0].value  # B的值
    heading_bin_centers = torch.from_numpy(np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN)).float().cuda()  # (NH,)
    """
    tf.constant(value,dtype=None,shape=None,name=’Const’)
    创建一个常量tensor，按照给出value来赋值，可以用shape来指定其形状。value可以是一个数，也可以是一个list。
    如果是一个数，那么这个常亮中所有值的按该数来赋值。
    如果是list,那么len(value)一定要小于等于shape展开后的长度。赋值时，先将value中的值逐个存入。不够的部分，则全部存入value的最后一个值。
    """
    size_residuals = size_residuals.permute(0,2,1)                         #为了兼容性  shape(B,3,NS) -> (B,NS,3)
    headings = heading_residuals + heading_bin_centers.unsqueeze(0)  # (B,NH)
    mean_sizes = torch.from_numpy(g_mean_size_arr).float().unsqueeze(0)+ size_residuals  # (1,NS,3)+(B,NS,3)
    #mean_sizes = torch.Tensor.repeat(torch.unsqueeze(torch.from_numpy(g_mean_size_arr).float(), 0).cuda() + size_residuals.cuda(), [batch_size,NUM_SIZE_CLUSTER,3])  # (B,NS,3)+(B,NS,3)
    sizes = mean_sizes + size_residuals  # (B,NS,3)
    sizes = sizes.unsqueeze(1).repeat(1, NUM_HEADING_BIN, 1, 1).float()  # (B,NH,NS,3)
    headings = headings.unsqueeze(2).repeat(1, 1, NUM_SIZE_CLUSTER)  # (B,NH,NS)
    centers = center.unsqueeze(1).unsqueeze(1).repeat(1, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1)  # (B,NH,NS,3)

    N = batch_size * NUM_HEADING_BIN * NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(torch.Tensor.view(centers, N, 3), torch.view(headings, N),torch.Tensor.view(sizes, N, 3))
    return corners_3d.view(batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3)


def huber_loss(error, delta=1.0):  # (B,), ()
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return torch.mean(losses)


def get_loss(mask_label, center_label, \
             heading_class_label, heading_residual_label, \
             size_class_label, size_residual_label, \
             end_points, \
             corner_loss_weight=10.0, \
             box_loss_weight=1.0):
    ''' Loss functions for 3D object detection.
    Input:
        mask_label:  int32 tensor in shape (B,N)
        center_label:  tensor in shape (B,3)
        heading_class_label:  int32 tensor in shape (B,)
        heading_residual_label:  tensor in shape (B,)
        size_class_label:  tensor int32 in shape (B,)
        size_residual_label: tensor tensor in shape (B,)
        end_points: dict, outputs from our model
        corner_loss_weight: float scalar
        box_loss_weight: float scalar
    Output:
        total_loss: scalar tensor
            the total_loss is also added to the losses collection
    '''
    # 3D Segmentation loss
    # tf.reduce_mean(input_tensor, reduction_indices=None)求平均值，
    # 若reduction_indices有值则在这个维度求均值
    mask_loss =  F.cross_entropy(end_points['mask_logits'],mask_label,reduction='mean')          # 计算一个 batch 的 mean_loss,  softmax + cross_entry_loss

    # Center regression losses
    #center_dist = torch.norm(center_label - end_points['center'], p=1, dim=1)                        # center_dist(B)
    center_loss = huber_loss(center_label - end_points['center'], delta=2.0)

    #stage1_center_dist = torch.norm(center_label - end_points['stage1_center'], dim=1)
    #stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
    stage1_center_loss = F.smooth_l1_loss(end_points['stage1_center'],center_label,reduction='mean')                  # 这里 的 smooth_l1 loss 的 参数为 1 相比于 center_loss的参数值为2 因为上一个的回归要求应该更加严格,所以参数较大离群项的惩罚更大。毕竟 center是 stage_center进一步计算回归得到的

    # Heading loss
    heading_class_loss = F.cross_entropy(end_points['heading_scores'], heading_class_label,reduction='mean')
    hcls_onehot = torch.eye(NUM_HEADING_BIN)[heading_class_label.long()]                                         #暂时删除了  .cuda() 原本为.cuda （B,NUM_CLASS）
    heading_residual_normalized_label = heading_residual_label / (np.pi / NUM_HEADING_BIN)
    #heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized']* torch.FloatTensor(hcls_onehot), dim=1).float() - heading_residual_normalized_label, delta=1.0)
    heading_residual_normalized_loss = F.smooth_l1_loss(torch.sum(end_points['heading_residuals_normalized']* torch.FloatTensor(hcls_onehot), dim=1).float(),heading_residual_normalized_label, reduction='mean')    #首先先把 heading 分到某个类，再进行回归，回归也是正则化之后回归。 正则化只 x 份 （pi/num_heading)

    # Size loss
    #size_class_loss = torch.mean(torch.nn.CrossEntropyLoss(end_points['size_scores'], size_class_label))
    size_class_loss = F.cross_entropy(end_points['size_scores'],size_class_label,reduction='mean')
    scls_onehot = torch.eye(NUM_SIZE_CLUSTER)[size_class_label.long()]                                     # BxNUM_SIZE_CLUSTER   先去掉.cuda
    # scls_onehot_tiled   = torch.Tensor.repeat(torch.Tensor.unsqueeze(scls_onehot.float(), dim=1), 1, 1, 3)   # BxNUM_SIZE_CLUSTERx3
    scls_onehot_tiled = scls_onehot.float().unsqueeze(1).repeat(1,3,1)
    # Bx1*NUM_SIZE_CLUSTER
    # Bx3xNUM_SIZE_CLUSTER
    predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized'] * scls_onehot_tiled, dim=2)  # Bx3
    mean_size_arr_expand = torch.from_numpy(g_mean_size_arr).float().cuda().unsqueeze(0)     # 1xNUM_SIZE_CLUSTERx3
    mean_size_arr_expand =  mean_size_arr_expand.permute(0,2,1)                                      # change shape to 1*3*NUN_SIZE_CLUSTER
    mean_size_label = torch.sum(scls_onehot_tiled * mean_size_arr_expand, dim=2)  # Bx3
    size_residual_label_normalized = size_residual_label / mean_size_label
    #size_normalized_dist = torch.norm(size_residual_label_normalized - predicted_size_residual_normalized, dim=1)
    #size_residual_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)
    size_residual_normalized_loss = F.smooth_l1_loss(predicted_size_residual_normalized,size_residual_label_normalized,reduction='mean')

    # Corner loss
    # We select the predicted corners corresponding to the
    # GT heading bin and size cluster.

    corners_3d = get_box3d_corners(end_points['center'], end_points['heading_residuals'], end_points['size_residuals'])             # (B,NH,NS,8,3)
    gt_mask = hcls_onehot.unsqueeze(2).repeat(1, 1, NUM_SIZE_CLUSTER) * scls_onehot.unsqueeze(1).repeat(1, NUM_HEADING_BIN, 1)  # (B,NH,NS)

    corners_3d_pred = torch.sum(gt_mask.unsqueeze(-1).unsqueeze(-1).float().cuda() * corners_3d, dim=[1, 2])  # (B,8,3)

    heading_bin_centers = torch.from_numpy(np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN)).float().cuda()  # (NH,)
    heading_label = heading_residual_label.unsqueeze(1) + heading_bin_centers.unsqueeze(0)  # (B,NH)

    heading_label = torch.sum(hcls_onehot.float() * heading_label, 1)
    mean_sizes = torch.from_numpy(g_mean_size_arr).float().view(1, NUM_SIZE_CLUSTER, 3).cuda()  # (1,NS,3)
    size_label = mean_sizes + size_residual_label.unsqueeze(1)  # (1,NS,3) + (B,1,3) = (B,NS,3)
    size_label = torch.sum(scls_onehot.float().unsqueeze(-1).float() * size_label, dim=1)  # (B,3)
    corners_3d_gt = get_box3d_corners_helper(center_label, heading_label, size_label)  # (B,8,3)
    corners_3d_gt_flip = get_box3d_corners_helper(center_label, heading_label + np.pi, size_label)  # (B,8,3)
    corners_dist = torch.clamp(torch.norm(corners_3d_pred - corners_3d_gt, dim=-1),
                               torch.norm(corners_3d_pred - corners_3d_gt_flip, dim=-1))
    corners_loss = huber_loss(corners_dist, delta=1.0)

    # Weighted sum of all losses
    total_loss = mask_loss + box_loss_weight * (center_loss +heading_class_loss + size_class_loss +heading_residual_normalized_loss * 20 +size_residual_normalized_loss * 20 + stage1_center_loss + corner_loss_weight * corners_loss)

    losses = {
        'total_loss': total_loss,
        'mask_loss': mask_loss,
        'mask_loss': box_loss_weight * center_loss,
        'heading_class_loss': box_loss_weight * heading_class_loss,
        'size_class_loss': box_loss_weight * size_class_loss,
        'heading_residual_normalized_loss': box_loss_weight * heading_residual_normalized_loss * 20,
        'size_residual_normalized_loss': box_loss_weight * size_residual_normalized_loss * 20,
        'stage1_center_loss': box_loss_weight * size_residual_normalized_loss * 20,
        'corners_loss': box_loss_weight * corners_loss * corner_loss_weight,
    }
    return losses['total_loss']
