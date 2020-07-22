from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
FPOINTNET_DIR = os.path.join(ROOT_DIR,'fpointnet')
SRCNN_DIR = os.path.join(ROOT_DIR,'srcnn')
sys.path.append(os.path.join(SRCNN_DIR,'lib'))
sys.path.append(os.path.join(FPOINTNET_DIR,'train'))

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.stereo_rcnn.resnet import resnet


#### fpointnet 相关
import fpointnet_trainer as FT
from train_util import get_batch_by_imgindex
import provider

ftrainer = FT.fpointnet_trainer()
TRAIN_DATASET = provider.FrustumDataset(npoints=ftrainer.config.NUM_POINT, split='train',
    rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True)

class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    #返回一个从0到训练集长度的随机整数排列
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)
    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

class TrainConfig():
    def __init__(self):
        self.start_epoch = 1
        self.epochs = 12
        self.save_dir = "models_stereo"
        self.num_workers = 1
        self.batch_size = 1
        self.lr_decay_step = 10
        self.lr_decay_gamma = 0.1
        self.resume = False
        self.checkepoch = 10
        self.checkepoint = 7421

class SrcnnTrainer():
    def __init__(self, config):
        self.start_epoch = config.start_epoch
        self.max_epochs = config.epochs
        self.epochs = config.epochs
        self.save_dir = config.save_dir
        self.num_workers = config.num_workers
        self.batch_size = config.batch_size
        self.lr_decay_step = config.lr_decay_step
        self.lr_decay_gamma = config.lr_decay_gamma
        self.resume = config.resume
        self.checkepoch = config.checkepoch
        self.checkepoint = config.checkepoint

    def train(self,device_id = 0):
        torch.cuda.set_device(device_id)
        print('Using config:')
        np.random.seed(cfg.RNG_SEED)

        imdb, roidb, ratio_list, ratio_index = combined_roidb('kitti_train')
        train_size = len(roidb)

        print('{:d} roidb entries'.format(len(roidb)))


        output_dir = self.save_dir + '/'
        if not os.path.exists(output_dir):
            print('save dir', output_dir)
            os.makedirs(output_dir)
        log_info = open((output_dir + 'trainlog.txt'), 'w')

        def log_string(out_str):
            log_info.write(out_str+'\n')
            log_info.flush()
            print(out_str)

        sampler_batch = sampler(train_size, self.batch_size)

        dataset = roibatchLoader(roidb, ratio_list, ratio_index, self.batch_size, \
                           imdb.num_classes, training=True)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                            sampler=sampler_batch, num_workers=self.num_workers)

        # initilize the tensor holder here.
        im_left_data = Variable(torch.FloatTensor(1).cuda())
        im_right_data = Variable(torch.FloatTensor(1).cuda())
        im_info = Variable(torch.FloatTensor(1).cuda())
        num_boxes = Variable(torch.LongTensor(1).cuda())
        gt_boxes_left = Variable(torch.FloatTensor(1).cuda())
        gt_boxes_right = Variable(torch.FloatTensor(1).cuda())
        gt_boxes_merge = Variable(torch.FloatTensor(1).cuda())
        gt_dim_orien = Variable(torch.FloatTensor(1).cuda())
        gt_kpts = Variable(torch.FloatTensor(1).cuda())
        pic_index = Variable(torch.LongTensor(1).cuda())

        # initilize the network here.
        stereoRCNN = resnet(imdb.classes, 101, pretrained=True)

        stereoRCNN.create_architecture()

        lr = cfg.TRAIN.LEARNING_RATE
        #lr = 0.0001

        uncert = Variable(torch.rand(6).cuda(), requires_grad=True)
        #torch.nn.init.constant(uncert, -1.0)
        torch.nn.init.constant_(uncert, -1.0)

        params = []
        for key, value in dict(stereoRCNN.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                    'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        params += [{'params':[uncert], 'lr':lr}]

        #初始化算梯度的玩意
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

        #分配显卡的使用device，不用管
        stereoRCNN.cuda()

        #初始化fpointent
        device = stereoRCNN.device
        ftrainer._init_model(device,"../fpointnet/train/log/models/fpointnet_all.pth")  # 读取预训练权重
        iters_per_epoch = int(train_size / self.batch_size)
        for epoch in range(self.start_epoch, self.max_epochs + 1):
            print('epoch: %d' % epoch)
            stereoRCNN.train()
            start = time.time()
            if epoch % (self.lr_decay_step + 1) == 0:
                adjust_learning_rate(optimizer, self.lr_decay_gamma)
                lr *= self.lr_decay_gamma

            data_iter = iter(dataloader)
            for step in range(iters_per_epoch):
                data = next(data_iter)

                im_left_data.resize_(data[0].size()).copy_(data[0])
                im_right_data.resize_(data[1].size()).copy_(data[1])
                im_info.resize_(data[2].size()).copy_(data[2])
                gt_boxes_left.resize_(data[3].size()).copy_(data[3])
                gt_boxes_right.resize_(data[4].size()).copy_(data[4])
                gt_boxes_merge.resize_(data[5].size()).copy_(data[5])
                gt_dim_orien.resize_(data[6].size()).copy_(data[6])
                gt_kpts.resize_(data[7].size()).copy_(data[7])
                num_boxes.resize_(data[8].size()).copy_(data[8])
                pic_index.resize_(data[9].size()).copy_(data[9])

                start = time.time()
                stereoRCNN.zero_grad()

                rois_left, rois_right, cls_prob, bbox_pred, gt_assign_left, dim_orien_pred, kpts_prob, \
                left_border_prob, right_border_prob, rpn_loss_cls, rpn_loss_box_left_right,\
                RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts, rois_label =\
                stereoRCNN(im_left_data, im_right_data, im_info, gt_boxes_left, gt_boxes_right, \
                    gt_boxes_merge, gt_dim_orien, gt_kpts, num_boxes)


                if (pic_index % 2) == 0:
                                pic_index = pic_index/2
                else:
                                pic_index = (pic_index-1)/2

                srcnn_output = {
                            'pic_index': pic_index,
                            'bbox_pred': bbox_pred,
                            'dim_orien_pred': dim_orien_pred,
                            'gt_assign_left': gt_assign_left,
                }
                print(bbox_pred.shape)
                print(dim_orien_pred.shape)
                print(gt_assign_left.shape)
                print()
                for index in range(16):
                            srcnn_output_index = {}
                            srcnn_output_index['pic_index'] = pic_index
                            srcnn_output_index['bbox_pred'] = bbox_pred[:,index*32:(index+1)*32,:]
                            srcnn_output_index['dim_orien_pred'] = dim_orien_pred[:,index*32:(index+1)*32,:]
                            srcnn_output_index['gt_assign_left'] = gt_assign_left[:,index*32:(index+1)*32]

                            ### compute fpointnet result
                print("print pic_index ")
                print(pic_index)
                print()
                print("print srcnn_output ")
                print(srcnn_output_index['bbox_pred'].shape)
                print()
                assert()




                optimizer.zero_grad()
                loss.backward()
                clip_gradient(stereoRCNN, 10.)
                optimizer.step()

                end = time.time()


            # 保存 fpointnet
            ftrainer._save_model()

            # 保存 srcnn
            save_name = os.path.join(output_dir, 'stereo_rcnn_{}_{}.pth'.format(epoch, step))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': stereoRCNN.state_dict(),
                'optimizer': optimizer.state_dict(),
                'uncert':uncert.data,
            }, save_name)

            log_string('save model: {}'.format(save_name))
            end = time.time()
            log_string('time %.4f' %(end - start))

if __name__ == '__main__':
    sconfig = TrainConfig();

    strainer = SrcnnTrainer(sconfig)
    strainer.train()
