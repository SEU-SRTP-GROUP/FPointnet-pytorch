import argparse
import importlib
import math
import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

#基本参数定义
from torch.autograd import Variable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #train文件夹的路径
ROOT_DIR = os.path.dirname(BASE_DIR)     #项目的根目录
sys.path.append(BASE_DIR)

#用到了 model provider 和 train_util
sys.path.append(os.path.join(ROOT_DIR, 'models'))
import provider
from train_util import get_batch

#超参数定义 不做修改

# 对于函数add_argument()参数第一个是选项，第二个是数据类型，第三个默认值，第四个是help命令时的说明
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
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

# Set training configurations

# 训练轮数
EPOCH_CNT = 0
# batch_size表明这个batch中包含多少个点云数据
BATCH_SIZE = FLAGS.batch_size
# num_point表明每个点云中含有多少个点
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
#   初始时使用较大的学习率较快地得到较优解，随着迭代学习率呈指数逐渐减小。
#  decayed_learning_rate = learning_rate*(decay_rate^(global_steps/decay_steps)
DECAY_RATE = FLAGS.decay_rate
NUM_CHANNEL = 3 if FLAGS.no_intensity else 4 # point feature channel
NUM_CLASSES = 2 # segmentation has two classes

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')  #模型的路径
LOG_DIR = FLAGS.log_dir                      #训练结果log的路径
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (os.path.join(BASE_DIR, 'train.py'), LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# Load Frustum Datasets. Use default data paths.
TRAIN_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='train',
    rotate_to_center=True, random_flip=True, random_shift=True, one_hot=True)
TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
    rotate_to_center=True, one_hot=True)


#pytorch 搭建神经网络

class Net (nn.Module):
    #init中定义卷积层
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

    #执行数据流动部分
    def forward(self, x):
        #x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x

    #line82 输出日志
    def log_string(out_str):
        LOG_FOUT.write(out_str + '\n')
        LOG_FOUT.flush()
        print(out_str)

    #line 87 获取指数衰减的learning rate
    def get_learning_rate(batch):
        #计算公式decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
        # 默认向下取整 可修改
        global_step=math.floor(batch * BATCH_SIZE)
        learning_rate=BASE_LEARNING_RATE*DECAY_RATE^(global_step/DECAY_STEP)
        learning_rate=torch.max(learning_rate,0.00001)
        # 原tf版：learing_rate = tf.maximum(learning_rate, 0.00001)  返回tensor
        return learning_rate

    #line100 指数衰减率
    def get_bn_decay(batch):
        global_step = math.floor(batch * BATCH_SIZE)
        bn_momentum=BN_INIT_DECAY*BN_DECAY_DECAY_RATE^(global_step/BN_DECAY_DECAY_RATE)
        bn_decay = torch.min(BN_DECAY_CLIP, 1 - bn_momentum)
        #tf版：bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        return bn_decay

    #将数据处理为variable
    def get_variable(x):
        x = Variable(x)
        return x.cuda() if torch.cuda.is_available() else x
    #line 110 开始训练
    def train(self):

        cnn=Net()
        #GPU加速
        if torch.cuda.is_available():
            cnn=cnn.cuda()

        # line 117-120
        #从module.util文件 line230 func:placeholder_inputs接收参数
        #需修改module.util func:placeholder_inputs函数 用到了tf.placeholder
        # 尝试改成torch.tensor
        '''
        pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
        heading_class_label_pl, heading_residual_label_pl, \
        size_class_label_pl, size_residual_label_pl = \
            MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        '''

        #
        is_training_pl = tf.placeholder(tf.bool, shape=())

        #获取model line 134
        #使用了 frustum_pointnets_v1.py line142 func:get_model 待修改
        end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
                                     is_training_pl, bn_decay=bn_decay)
        #指定gpu line-116
        device = torch.device("cuda:1")

        #loss function 调用model中的loss function line 136
        #module.util line358 func:get_loss 需修改
        loss = MODEL.get_loss(labels_pl, centers_pl,
                              heading_class_label_pl, heading_residual_label_pl,
                              size_class_label_pl, size_residual_label_pl, end_points)

        #Optimizer
        # Get training operator
        learning_rate = self.get_learning_rate(batch)
        if OPTIMIZER == 'momentum':
            optimizer = torch.optim.SGD(learning_rate,momentum=MOMENTUM)
        elif OPTIMIZER == 'adam':
            optimizer = torch.optim.Adam(learning_rate)

        #batch训练
   








