import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from train import provider
from train.train_util import  get_batch
import numpy as np
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util import g_type_mean_size,g_mean_size_arr
from   model.model_util import get_loss
import torch.nn.functional as F


def convert_to_one_hot(num_class,non_one_hot,*,dim):
    '''
    @chonepieceyb 将非 onehot 编码 转化为 one hot 编码
    :param num_class:    类别数
    :param non_one_hot:  (di,di2... 1 ... d3 d4)
    :param dim  alone which dim
    :return:  (di,di2.... num_class d3 ,d 4)
    '''
    shape = list(non_one_hot.shape)
    shape.append(num_class)   # 原始的
    one_hot = np.eye(num_class)[non_one_hot.reshape(-1)].reshape(shape)
    dims = list(range(len(shape)))
    dims[dim] , dims[-1] = dims[-1], dims[dim]
    return one_hot.transpose(*dims)


class TestingUtil(object):
    def __init__(self):
        self.NUM_POINT = 2048
        self.DATASET =  provider.FrustumDataset(npoints=self.NUM_POINT, split='train',
        rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True)
        self.NUM_CHANNEL = 4         #加上雷达强度
        self.normal_mean = 0         #正态分布的均值
        self.normal_var =1           #正态分布的方差
        self.negative_low = 0
        self.negative_high = 0.3
        self.positive_low = 0.7
        self.positive_high = 1.0
        self.shift = 0.3           #回归偏置
        self.class_low = 0       #分类偏置  0.1 / 0.9
        self.class_high =1

    def get_batch_data(self,batch_size,batch_index = 0 ,shuffled =False,*, add_noise=False , datyType ="np",device ='cpu'):
        '''
        @ author chonepieceyb 从数据集中获取 数据
        :param batch_size:  batch 的数目
        :param batch_index: 从第几个 batch 开始取
        :param suffled:  每次取的时候是否打乱顺序
        :param datyType:  如果是 "torch" 返回 在 device 上的 torch tensor , 如果是 "numpy" 返回 numpy数组
        :return: data a dict ,data format is torch style
                data {
                     batch_data : (B,C,N)
                     mask_label:   (B,N)   dim=1  1： 选中 0:不选中
                     heading_class_label: 分别属于哪一种heading的得分   shape(B,)
                     heading_residuals_normalized_label: 正则化后的heading 残差项 : shape(B,)
                     heading_residuals_label  : 未正则化项  shape(B,)
                     size_class_label      :  分别属于哪一种物体的大小（car...） 的得分    shape(B,)
                     size_residuals_normalized_label ： 正则化后的box size 残差项         shape(B,3)
                     size_residuals_label:     未正则化项                                shape(B,3)
                     center_label            : 全局坐标             （B,3)
                }
                end_points
                {
                     mask_logits : tensor in shape (B,2,N)
                     mask:   (B,N)   dim=1  1： 选中 0:不选中  和 mask_label完全相同
                     stage1_center : 每一个proposal的估计的中心坐标  shape (B,3)
                     center_boxnet : box回归网络中，预测得到的局部坐标系坐标  shape (B,3)
                     heading_scores: 分别属于哪一种heading的得分             shape(B,NUM_HEADING_BIN)
                     heading_residuals_normalized: 正则化后的heading 残差项 :        shape(B,NUM_HEADING_BIN)
                     heading_residuals   : 未正则化项  shape(B,NUM_HEADING_BIN)
                     size_scores      :  分别属于哪一种物体的大小（car...） 的得分    shape(B,NUM_SIZE_CLUSTER)
                     size_residuals_normalized ： 正则化后的box size 残差项         shape(B,3,NUM_SIZE_CLUSTER)
                     size_residuals:     未正则化项
                     center            : 全局坐标                               （B,3)
                }
        '''
        train_index = np.arange(0,len(self.DATASET))
        if  shuffled :
            np.random.shuffle(train_index)  # 随机
        num_batches = len(self.DATASET) // batch_size
        batch_index = batch_index if batch_index < num_batches else 0
        start_idx = batch_index * batch_size
        end_idx = (batch_index+1) * batch_size
        # 获得 原始数据
        '''
        batch_data = np.zeros((bsize, num_point, num_channel))
        batch_label = np.zeros((bsize, num_point), dtype=np.int32)
        batch_center = np.zeros((bsize, 3))
        batch_heading_class = np.zeros((bsize,), dtype=np.int32)
        batch_heading_residual = np.zeros((bsize,))
        batch_size_class = np.zeros((bsize,), dtype=np.int32)
        batch_size_residual = np.zeros((bsize, 3))
        batch_rot_angle = np.zeros((bsize,))
        '''
        data = {}
        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = \
            get_batch(self.DATASET,    train_index, start_idx, end_idx,
                      self.NUM_POINT, self.NUM_CHANNEL)
        batch_data = batch_data.transpose(0,2,1)                               # change data format
        data["batch_data"] = batch_data
        data["center_label"] = batch_center
        data["mask_label"]  = batch_label
        data["heading_class_label"] = batch_hclass
        data["heading_residuals_label"] =  batch_hres
        data["heading_residuals_normalized_label"] = batch_hres/ (np.pi / NUM_HEADING_BIN)
        data["size_class_label"]  = batch_sclass                   # shape (B,)
        data["size_residuals_label"] = batch_sres

        mean_size_arr = g_mean_size_arr.transpose(1,0)    # change data format
        mean_size_arr = np.expand_dims(mean_size_arr,axis=0)   # shape ( 1, 3, NC)
        size_label_onehot  = convert_to_one_hot(NUM_SIZE_CLUSTER, batch_sclass, dim=1)  # shape( B,NC)
        size_label_onehot_tailed= np.tile(np.expand_dims(size_label_onehot,axis=1),(1,3,1))   # shape(B,3,NC)
        mean_size_label = np.sum(size_label_onehot_tailed* mean_size_arr,axis=2,keepdims=False)  # shape(B,3)
        data["size_residuals_normalized_label"] = batch_sres/mean_size_label

        '''
        mean_size_arr_expand = torch.from_numpy(g_mean_size_arr).float().unsqueeze(0).to(device)     # 1xNUM_SIZE_CLUSTERx3
        mean_size_arr_expand =  mean_size_arr_expand.permute(0,2,1)                                      # change shape to 1*3*NUN_SIZE_CLUSTER
        mean_size_label = torch.sum(scls_onehot_tiled * mean_size_arr_expand, dim=2).to(device)  # Bx3
        size_residual_label_normalized = size_residual_label / mean_size_label
        size_residual_normalized_loss = F.smooth_l1_loss(predicted_size_residual_normalized,size_residual_label_normalized,reduction='mean')
        '''
        end_points ={}
        # 给原始数据添加噪声 来模拟生成的数据..
        # mask
        # 使用向量化函数
        # 采用 均匀分布 来获取mask_logist



        #计算随机center
        if add_noise == True:
            func = lambda x: np.random.uniform(self.negative_low, self.negative_high) if x < 0.5 else np.random.uniform(self.positive_low, self.positive_high)
            vfun = np.vectorize(func, otypes=[np.float32])
            mask_logits = convert_to_one_hot(2, batch_label, dim=1)
            mask_logits = vfun(mask_logits)
            end_points["mask_logits"] = mask_logits
            end_points["mask"] = batch_label  # 这里注意

            end_points['center'] = batch_center + np.random.normal(self.normal_mean,self.normal_var,batch_center.shape)     # 给 center标签加上高斯噪声，获取 “预测的 center"
            center_boxnet = np.random.normal(self.normal_mean,self.normal_var,batch_center.shape)               # 用 高斯噪声 假设为 预测的残差
            end_points['center_boxnet'] = center_boxnet        # 这个应该允许负值，因为已经在局部坐标系里了
            end_points['stage1_center'] = end_points['center'] - center_boxnet

            # 计算随机的 heading
            angle_per_class = 2 * np.pi / NUM_HEADING_BIN
            heading_scores_one_hot = convert_to_one_hot(NUM_HEADING_BIN, data['heading_class_label'], dim=1)  # (B, NC)
            end_points["heading_scores"] = vfun(heading_scores_one_hot)
            heading_residuals = np.tile(np.expand_dims(batch_hres, axis=1), (1, NUM_HEADING_BIN))  # (B,NC)
            heading_residuals = (heading_residuals + (angle_per_class / 2) * (
                np.random.normal(self.normal_mean, self.normal_var, heading_residuals.shape))) % (
                                            angle_per_class / 2)  # (B,NC)添加高斯噪声,待研究
            heading_residuals = heading_residuals * heading_scores_one_hot  # (B,NC)   屏蔽掉 不是 1 的 ，因为 在损失函数中 只有正确的项 有贡献
            end_points["heading_residuals"] = heading_residuals
            end_points["heading_residuals_normalized"] = heading_residuals / (np.pi / NUM_HEADING_BIN)

            # 计算随机的size
            end_points["size_scores"] = vfun(size_label_onehot)  # (B,NC)
            size_residuals = np.tile(np.expand_dims(batch_sres, axis=2),
                                     (1, 1, NUM_SIZE_CLUSTER))  # (B,3,1) -> (B,3,NC)
            size_residuals = size_residuals + np.random.normal(self.normal_mean, self.normal_var,
                                                               size_residuals.shape)  # 添加高斯噪声   (B,3,NC)
            size_residuals = size_residuals * size_label_onehot_tailed  # (B,3,NC)   屏蔽掉 不是 1 的 ，因为 在损失函数中 只有正确的项 有贡献
            end_points["size_residuals"] = size_residuals
            end_points["size_residuals_normalized"] = size_residuals / mean_size_arr
        else:
            func = lambda x:self.class_low if x < 0.5 else self.class_high
            vfun = np.vectorize(func, otypes=[np.float32])
            mask_logits = convert_to_one_hot(2, batch_label, dim=1)
            mask_logits = vfun(mask_logits)
            end_points["mask_logits"] = mask_logits
            end_points["mask"] = batch_label  # 这里注意

            end_points['center'] = batch_center + self.shift # 给 center标签加上高斯噪声，获取 “预测的 center"
            center_boxnet = np.ones_like(batch_center) * self.shift  # 用 高斯噪声 假设为 预测的残差
            end_points['center_boxnet'] = center_boxnet  # 这个应该允许负值，因为已经在局部坐标系里了
            end_points['stage1_center'] = end_points['center'] - center_boxnet

            # 计算随机的 heading
            angle_per_class = 2 * np.pi / NUM_HEADING_BIN
            heading_scores_one_hot = convert_to_one_hot(NUM_HEADING_BIN, data['heading_class_label'], dim=1)  # (B, NC)
            end_points["heading_scores"] = vfun(heading_scores_one_hot)
            heading_residuals = np.tile(np.expand_dims(batch_hres, axis=1), (1, NUM_HEADING_BIN))  # (B,NC)
            heading_residuals = (heading_residuals + (angle_per_class / 2) * (
                self.shift))  # (B,NC)添加高斯噪声,待研究
            heading_residuals = heading_residuals * heading_scores_one_hot  # (B,NC)   屏蔽掉 不是 1 的 ，因为 在损失函数中 只有正确的项 有贡献
            end_points["heading_residuals"] = heading_residuals
            end_points["heading_residuals_normalized"] = heading_residuals / (np.pi / NUM_HEADING_BIN)

            # 计算随机的size
            end_points["size_scores"] = vfun(size_label_onehot)  # (B,NC)
            size_residuals = np.tile(np.expand_dims(batch_sres, axis=2),
                                     (1, 1, NUM_SIZE_CLUSTER))  # (B,3,1) -> (B,3,NC)
            size_residuals = size_residuals + self.shift # 添加高斯噪声   (B,3,NC)
            size_residuals = size_residuals * size_label_onehot_tailed  # (B,3,NC)   屏蔽掉 不是 1 的 ，因为 在损失函数中 只有正确的项 有贡献
            end_points["size_residuals"] = size_residuals
            end_points["size_residuals_normalized"] = size_residuals / mean_size_arr


        if datyType =="torch" :
            for key , value in data.items():
                data[key] = torch.from_numpy(value).to(device)
            for key, value in end_points.items():
                end_points[key] = torch.from_numpy(value).to(device).type(torch.float32)
            return data , end_points
        else:
            for key, value in end_points.items():
                end_points[key] = value.astype(np.float32)
            return data , end_points


if __name__ == '__main__':
    testing = TestingUtil()
    data,end_points = testing.get_batch_data(1,0,datyType='torch')
    '''
    mask_label, center_label, \
             heading_class_label, heading_residual_label, \
             size_class_label, size_residual_label
    '''
    print('############### center_label ####################')
    print(data['center_label'])
    print(end_points['center'])
    print(end_points['stage1_center'])
    print(end_points['center_boxnet'])

    print("############### heading class ###################")
    print(data['heading_class_label'])
    print(end_points['heading_scores'])

    print("############### heading_residuals ###################")
    print(data['heading_residuals_label'])
    print(end_points['heading_residuals'])

    print("############### size_class ###################")
    print(data['size_class_label'])
    print(end_points['size_scores'])

    print("############### size_residuals ###################")
    print(data['size_residuals_label'])
    print(end_points['size_residuals'])

    for value in end_points.values():
        print(value.dtype)

    total,losses = get_loss(data['mask_label'].type(torch.LongTensor), data['center_label'].type(torch.float32),
             data['heading_class_label'].type(torch.LongTensor), data['heading_residuals_label'].type(torch.float32),
             data['size_class_label'].type(torch.LongTensor)
             , data['size_residuals_label'].type(torch.float32), end_points)
    for key , value in losses.items():
        print(key, value)

