import argparse
import importlib
import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import model
import provider
from model.frustum_pointnets_v1 import FPointNet
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
# parse_args() 传递一组参数字符串来解析命令行，返回一个命名空间包含传递给命令的参数
FLAGS = parser.parse_args()

#训练参数
EPOCH_CNT = 0
# batch_size表明这个batch中包含多少个点云数据
BATCH_SIZE = FLAGS.batch_size
# num_point表明每个点云中含有多少个点
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
# GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
#   初始时使用较大的学习率较快地得到较优解，随着迭代学习率呈指数逐渐减小。
#  decayed_learning_rate = learning_rate*(decay_rate^(global_steps/decay_steps)
DECAY_RATE = FLAGS.decay_rate
NUM_CHANNEL = 3 if FLAGS.no_intensity else 4 # point feature channel
NUM_CLASSES = 2 # segmentation has two classes
LOG_DIR = FLAGS.log_dir                      #训练结果log的路径

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

################################网络参数##################################

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Load Frustum Datasets. Use default data paths.
TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='train',
    rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True)
TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
    rotate_to_center=True, one_hot=True)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    global_step = int(batch * BATCH_SIZE)
    learning_rate = BASE_LEARNING_RATE*DECAY_RATE**(global_step/DECAY_STEP)
    learing_rate = torch.max(learning_rate, 0.00001)
    return learning_rate

def get_bn_decay(batch):
    global_step = int(batch * BATCH_SIZE)
    bn_momentum = BN_INIT_DECAY*BN_DECAY_DECAY_RATE**(global_step/BN_DECAY_DECAY_STEP)
    bn_decay = torch.min(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

#可以把get_batch拿到的直接可并进来其实，后面再改
def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = Variable(torch.FloatTensor(batch_size, num_point, 4).zero_().cuda())
    one_hot_vec_pl = Variable(torch.FloatTensor(batch_size, 3).zero_().cuda())

    # labels_pl is for segmentation label
    labels_pl = Variable(torch.FloatTensor(batch_size, num_point).zero_().cuda())
    centers_pl = Variable(torch.FloatTensor(batch_size, 3).zero_().cuda())
    heading_class_label_pl = Variable(torch.IntTensor(batch_size,).zero_().cuda())
    heading_residual_label_pl = Variable(torch.FloatTensor(batch_size,).zero_().cuda())
    size_class_label_pl = Variable(torch.IntTensor(batch_size,).zero_().cuda())
    size_residual_label_pl = Variable(torch.FloatTensor(batch_size, 3).zero_().cuda())

    return pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
        heading_class_label_pl, heading_residual_label_pl, \
        size_class_label_pl, size_residual_label_pl

def train():
    fpointnet = FPointNet()

    pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
    heading_class_label_pl, heading_residual_label_pl, \
    size_class_label_pl, size_residual_label_pl = \
        placeholder_inputs(BATCH_SIZE, NUM_POINT)

    is_training_pl = Variable(torch.BoolTensor(1).cuda())

    batch = Variable(torch.zeros(1).cuda(), requires_grad=False)
    bn_decay = get_bn_decay(batch)

    # Get model and losses
    # end_points = fpointnet.forward(pointclouds_pl, one_hot_vec_pl)
    # loss = fpointnet.get_loss(labels_pl, centers_pl,
    #             heading_class_label_pl, heading_residual_label_pl,
    #             size_class_label_pl, size_residual_label_pl, end_points)

    #初始优化器,只放需要训练的参数，需再调整
    params = []
    learning_rate = get_learning_rate(batch)
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    fpointnet = FPointNet()

    for epoch in range(MAX_EPOCH):
        print('epoch: %d' % epoch)
        is_training = True
        log_string(str(datetime.now()))

        # Shuffle train samples
        # 对dataset进行随机排列打乱顺序，不知道用途是干啥的大概是符合高斯分布的随机数
        train_idxs = np.arange(0, len(TRAIN_DATASET))
        np.random.shuffle(train_idxs)
        num_batches = len(TRAIN_DATASET)//BATCH_SIZE

        # To collect statistics
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        iou2ds_sum = 0
        iou3ds_sum = 0
        iou3d_correct_cnt = 0

        # Training with batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            batch_data, batch_label, batch_center, \
            batch_hclass, batch_hres, \
            batch_sclass, batch_sres, \
            batch_rot_angle, batch_one_hot_vec = \
            get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)

            pointclouds_pl = batch_data
            one_hot_vec_pl = batch_one_hot_vec
            labels_pl = batch_label
            centers_pl = batch_center
            heading_class_label_pl = batch_hclass
            heading_residual_label_pl = batch_hres
            size_class_label_pl = batch_sclass
            size_residual_label_pl = batch_sres

            fpointnet.train()

            stereoRCNN.zero_grad()
            end_points = fpointnet.forward(pointclouds_pl, one_hot_vec_pl)

            loss = fpointnet.get_loss(labels_pl, centers_pl,\
                  heading_class_label_pl, heading_residual_label_pl,\
                  size_class_label_pl, size_residual_label_pl, end_points)

if __name__ == '__main__':
    train()
