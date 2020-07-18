"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
import os
import sys
import numpy as np
import torch
from torch.autograd import Variable
from model.utils.config import cfg
from datasets.factory import get_imdb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from torch.utils.data.sampler import Sampler
import PIL
import pdb

def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """

  roidb = imdb.roidb
  if not (imdb.name.startswith('coco')):
    sizes = [PIL.Image.open(imdb.img_left_path_at(i)).size
         for i in range(imdb.num_images)]
         
  for i in range(len(imdb.image_index)):
    roidb[i]['img_left'] = imdb.img_left_path_at(i)
    roidb[i]['img_right'] = imdb.img_right_path_at(i)
    if not (imdb.name.startswith('coco')):
      roidb[i]['width'] = sizes[i][0]
      roidb[i]['height'] = sizes[i][1]
    # need gt_overlaps as a dense array for argmax
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)
    roidb[i]['max_classes'] = max_classes
    roidb[i]['max_overlaps'] = max_overlaps
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    nonzero_inds = np.where(max_overlaps > 0)[0]
    assert all(max_classes[nonzero_inds] != 0)


def rank_roidb_ratio(roidb):
    '''
    对roidb的图片按照长宽比进行排序标记
    :param roidb:
    :return: ratio_list：roidb中每张图片的ratio
              ratio_index：对应ratio_list中每个ratio在list中的排序
    '''
    # rank roidb based on the ratio between width and height.
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.    
    
    ratio_list = []
    for i in range(len(roidb)):
      width = roidb[i]['width']
      height = roidb[i]['height']
      ratio = width / float(height)

      if ratio > ratio_large:
        roidb[i]['need_crop'] = 1
        ratio = ratio_large
      elif ratio < ratio_small:
        roidb[i]['need_crop'] = 1
        ratio = ratio_small        
      else:
        roidb[i]['need_crop'] = 0

      ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    '''
    洗掉没有标记的图，减少无用数据量
    :param roidb:
    :return:
    '''
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
      if len(roidb[i]['boxes_left']) == 0 or len(roidb[i]['boxes_right']) == 0 or len(roidb[i]['boxes_merge']) == 0:
        del roidb[i]
        i -= 1
      i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb

def combined_roidb(imdb_names, training=True):
  """
  Combine multiple roidbs
  """
  def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
      print('Appending horizontally-flipped training examples...')
      imdb.append_flipped_images()
      print('done')

    print('Preparing training data...')

    prepare_roidb(imdb)
    #ratio_index = rank_roidb_ratio(imdb)
    print('done')

    return imdb.roidb
  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print(imdb.num_images)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method('gt')
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]

  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)

  if training:
    roidb = filter_roidb(roidb)

  ratio_list, ratio_index = rank_roidb_ratio(roidb)

  return imdb, roidb, ratio_list, ratio_index

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
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':
  imdb, roidb, ratio_list, ratio_index = combined_roidb('kitti_train')
  train_size = len(roidb)
  batch_size = 1
  num_point = 1024
  sampler_batch = sampler(train_size, batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, \
                           imdb.num_classes,num_point, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler_batch, num_workers=1)

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

  #fpointnet data
  point_set = Variable(torch.FloatTensor(1).cuda())
  seg = Variable(torch.FloatTensor(1).cuda())
  box3d_center = Variable(torch.FloatTensor(1).cuda())
  angle_class = Variable(torch.FloatTensor(1).cuda())
  angle_residual = Variable(torch.FloatTensor(1).cuda())
  size_class = Variable(torch.FloatTensor(1).cuda())
  size_residual = Variable(torch.FloatTensor(1).cuda())
  rot_angle = Variable(torch.FloatTensor(1).cuda())
  one_hot_vec = Variable(torch.FloatTensor(1).cuda())

  data_iter = iter(dataloader)
  for step in range(5):
      data = next(data_iter)
      #print(data[8].size())

      im_left_data.resize_(data[0].size()).copy_(data[0])
      im_right_data.resize_(data[1].size()).copy_(data[1])
      im_info.resize_(data[2].size()).copy_(data[2])
      gt_boxes_left.resize_(data[3].size()).copy_(data[3])
      gt_boxes_right.resize_(data[4].size()).copy_(data[4])
      gt_boxes_merge.resize_(data[5].size()).copy_(data[5])
      gt_dim_orien.resize_(data[6].size()).copy_(data[6])
      gt_kpts.resize_(data[7].size()).copy_(data[7])
      num_boxes.resize_(data[8].size()).copy_(data[8])
      print()
      point_set.resize_(data[9].size()).copy_(data[9])
      seg.resize_(data[10].size()).copy_(data[10])
      box3d_center.resize_(data[11].size()).copy_(data[11])
      angle_class.resize_(data[12].size()).copy_(data[12])
      angle_residual.resize_(data[13].size()).copy_(data[13])
      size_class.resize_(data[14].size()).copy_(data[14])
      size_residual.resize_(data[15].size()).copy_(data[15])
      rot_angle.resize_(data[16].size()).copy_(data[16])
      one_hot_vec.resize_(data[17].size()).copy_(data[17])

      assert()
