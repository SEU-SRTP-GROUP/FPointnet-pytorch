
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch
import cv2
import os
import sys
import pickle

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
SRCNN_DIR = os.path.join(ROOT_DIR,'srcnn')
# sys.path.append(os.path.join(SRCNN_DIR,'lib'))
sys.path.append(SRCNN_DIR)

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch, get_minibatch
from roi_data_layer.model_util import g_type2class, g_class2type, g_type2onehotclass
from roi_data_layer.model_util import g_type_mean_size
from roi_data_layer.model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER

from roi_data_layer.provider import FrustumDataset
# from bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb

class roibatchLoader(data.Dataset):
  '''
    对roidb进行迭代，重写getitem函数方便索引
    类内可包含FrustumDataset同时读取点云视锥
    需要点云视锥的对应可能还需要调整不能直接getitem获得
  '''
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
    self._roidb = roidb
    self._num_classes = num_classes
    self.max_num_box = cfg.MAX_NUM_GT_BOXES   #30
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    self.data_size = len(self.ratio_list)
    self.num_channel = 4
    self.num_point = 1024
    self.TRAIN_DATASET = FrustumDataset(npoints=self.num_point, split='train',
    rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True)
    self.TEST_DATASET = FrustumDataset(npoints=self.num_point, split='val',
    rotate_to_center=True, one_hot=True)

  def __getitem__(self, index):
    if self.training:
        index_ratio = int(self.ratio_index[index])
    else:
        index_ratio = index

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index_ratio]]
    blobs = get_minibatch(minibatch_db, self._num_classes)

    data_left = torch.from_numpy(blobs['data_left'])
    data_right = torch.from_numpy(blobs['data_right'])
    im_info = torch.from_numpy(blobs['im_info'])

    if self.training:
        boxes_all = np.concatenate((blobs['gt_boxes_left'], blobs['gt_boxes_right'],blobs['gt_boxes_merge'],\
                                    blobs['gt_dim_orien'], blobs['gt_kpts']), axis=1)
        
        np.random.shuffle(boxes_all) 

        gt_boxes_left = torch.from_numpy(boxes_all[:,0:5]) #[N,(x1, y1, x2, y2, cls)]
        gt_boxes_right = torch.from_numpy(boxes_all[:,5:10])    #[N,(x1, y1, x2, y2, cls)]
        gt_boxes_merge = torch.from_numpy(boxes_all[:,10:15])   #[N,(x1, y1, x2, y2, cls)]
        gt_dim_orien = torch.from_numpy(boxes_all[:,15:20]) #[N,(x1, y1, x2, y2, cls)]
        gt_kpts = torch.from_numpy(boxes_all[:,20:26])  #[N,(keypoints*4,visible_left,visible_right)]

        num_boxes = min(gt_boxes_left.size(0), self.max_num_box) 

        data_left = data_left[0].permute(2, 0, 1).contiguous()
        data_right = data_right[0].permute(2, 0, 1).contiguous()
        im_info = im_info.view(3) 

        gt_boxes_left_padding = torch.FloatTensor(self.max_num_box, gt_boxes_left.size(1)).zero_()
        gt_boxes_left_padding[:num_boxes,:] = gt_boxes_left[:num_boxes]
        gt_boxes_left_padding = gt_boxes_left_padding.contiguous()

        gt_boxes_right_padding = torch.FloatTensor(self.max_num_box, gt_boxes_right.size(1)).zero_()
        gt_boxes_right_padding[:num_boxes,:] = gt_boxes_right[:num_boxes]
        gt_boxes_right_padding = gt_boxes_right_padding.contiguous() 

        gt_boxes_merge_padding = torch.FloatTensor(self.max_num_box, gt_boxes_merge.size(1)).zero_()
        gt_boxes_merge_padding[:num_boxes,:] = gt_boxes_merge[:num_boxes]
        gt_boxes_merge_padding = gt_boxes_merge_padding.contiguous()

        gt_dim_orien_padding = torch.FloatTensor(self.max_num_box, gt_dim_orien.size(1)).zero_()
        gt_dim_orien_padding[:num_boxes,:] = gt_dim_orien[:num_boxes]
        gt_dim_orien_padding = gt_dim_orien_padding.contiguous()

        gt_kpts_padding = torch.FloatTensor(self.max_num_box, gt_kpts.size(1)).zero_()
        gt_kpts_padding[:num_boxes,:] = gt_kpts[:num_boxes]
        gt_kpts_padding = gt_kpts_padding.contiguous()

        #直接index获取视锥，如需要图片和视锥对应可以通过类似get_batch中get_batch
        #将batch_size个视锥封装为对应同一张图片的batch中
        ps,seg,center,hclass,hres,sclass,sres,rotangle,onehotvec = \
                self.TRAIN_DATASET[index]

        return data_left, data_right, im_info, gt_boxes_left_padding, gt_boxes_right_padding,\
               gt_boxes_merge_padding, gt_dim_orien_padding, gt_kpts_padding, num_boxes,index_ratio,\
               ps,seg,center,hclass,hres,sclass,sres,rotangle,onehotvec
    else: 
        data_left = data_left[0].permute(2, 0, 1).contiguous()
        data_right = data_right[0].permute(2, 0, 1).contiguous()
        im_info = im_info.view(3) 


        gt_boxes = torch.FloatTensor([1,1,1,1,1])
        num_boxes = 0

        return data_left, data_right, im_info, gt_boxes, gt_boxes, gt_boxes, gt_boxes, gt_boxes, num_boxes,\


  def __len__(self):
    return len(self._roidb)
