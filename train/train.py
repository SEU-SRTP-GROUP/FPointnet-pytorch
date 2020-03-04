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
import torch.nn.functional as F

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
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

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

            # 指定gpu line-116
            device = torch.device("cuda:1")

            # line 117-120
            #从module.util文件 line230 func:placeholder_inputs接收参数
            #需修改module.util func:placeholder_inputs函数 用到了tf.placeholder
            # 改成用torch.Variable封装
            #---------
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            #---------

            #传入cnn shape:(B,4,N) (B,3)
            out=cnn(pointclouds_pl,one_hot_vec_pl)
            out.size()
            #反向传播前清零梯度
            cnn.zero_grad()
            out.backward()

            #是否正在训练
            is_training_pl=Variable(torch.BoolTensor(1).cuda())

            #batch = tf.get_variable('batch', [],initializer=tf.constant_initializer(0), trainable=False)
            batch=torch.zeros([],'batch')
            bn_decay = self.get_bn_decay(batch)

            # 用于tensorboard显示训练过程，scalar用来显示标量信息
            # tf.summary.scalar('bn_decay', bn_decay)


            # 获取model line 134
            # 使用了 frustum_pointnets_v1.py line142 func:get_model 待修改
            #---------
            end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
                                         is_training_pl, bn_decay=bn_decay)
            #---------

            # loss function 调用model中的loss function line 136
            # module.util line358 func:get_loss 需修改
            #---------
            loss = MODEL.get_loss(labels_pl, centers_pl,
                                  heading_class_label_pl, heading_residual_label_pl,
                                  size_class_label_pl, size_residual_label_pl, end_points)
            # ---------


            # tf.get_collection：从一个集合中取出全部变量，是一个列表
            # tf.add_n：把一个列表的东西都依次加起来
            #上下文都没用到losses？ 好像是用来画benchmark的先不管
            '''
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            '''

            #计算box3d_iou line-150
            iou2ds, iou3ds=provider.compute_box3d_iou([\
                end_points['center'], \
                end_points['heading_scores'], end_points['heading_residuals'], \
                end_points['size_scores'], end_points['size_residuals'], \
                centers_pl, \
                heading_class_label_pl, heading_residual_label_pl, \
                size_class_label_pl, size_residual_label_pl])

            iou2ds=torch.from_numpy(iou2ds)
            iou3ds=torch.from_numpy(iou3ds)
            end_points['iou2ds'] = iou2ds
            end_points['iou3ds'] = iou3ds

            correct=torch.equal(torch.argmax(end_points['mask_logits'], 2),labels_pl.to(torch.long))
            accuracy=torch.sum(correct.to(torch.float32))/float(BATCH_SIZE*NUM_POINT)
            learning_rate = self.get_learning_rate(batch)
            #Optimizer
            # Get training operator
            learning_rate = self.get_learning_rate(batch)
            if OPTIMIZER == 'momentum':
                optimizer = torch.optim.SGD(learning_rate,momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = torch.optim.Adam(learning_rate)
            #optim



            #pytorch结构有待调整
            #line180
    def train_one_epoch( ops, train_writer):
        is_training = True
        log_string(str(datetime.now()))
        # Shuffle train samples
        # 对dataset进行随机排列打乱顺序，不知道用途是干啥的大概是符合高斯分布的随机数
        train_idxs = np.arange(0, len(TRAIN_DATASET))
        np.random.shuffle(train_idxs)
        num_batches = len(TRAIN_DATASET) // BATCH_SIZE

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
            end_idx = (batch_idx + 1) * BATCH_SIZE

            #train.util---get_batch
            batch_data, batch_label, batch_center, \
            batch_hclass, batch_hres, \
            batch_sclass, batch_sres, \
            batch_rot_angle, batch_one_hot_vec = \
                get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx,
                          NUM_POINT, NUM_CHANNEL)
            # feed_dict对占位符placeholder传入数据
            #修改train.util

            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['one_hot_vec_pl']: batch_one_hot_vec,
                         ops['labels_pl']: batch_label,
                         ops['centers_pl']: batch_center,
                         ops['heading_class_label_pl']: batch_hclass,
                         ops['heading_residual_label_pl']: batch_hres,
                         ops['size_class_label_pl']: batch_sclass,
                         ops['size_residual_label_pl']: batch_sres,
                         ops['is_training_pl']: is_training, }
            summary, step, _, loss_val, logits_val, centers_pred_val, \
            iou2ds, iou3ds = \
                         [ops['merged'], ops['step'], ops['train_op'], ops['loss'],
                          ops['logits'], ops['centers_pred'],
                          ops['end_points']['iou2ds'], ops['end_points']['iou3ds']]
            # 把这一轮训练的结果返回修正参数
            preds_val = np.argmax(logits_val, 2)
            correct = np.sum(preds_val == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss_val
            iou2ds_sum += np.sum(iou2ds)
            iou3ds_sum += np.sum(iou3ds)
            iou3d_correct_cnt += np.sum(iou3ds >= 0.7)

            if (batch_idx + 1) % 10 == 0:
                log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
                log_string('mean loss: %f' % (loss_sum / 10))
                log_string('segmentation accuracy: %f' % \
                           (total_correct / float(total_seen)))
                log_string('box IoU (ground/3D): %f / %f' % \
                           (iou2ds_sum / float(BATCH_SIZE * 10), iou3ds_sum / float(BATCH_SIZE * 10)))
                log_string('box estimation accuracy (IoU=0.7): %f' % \
                           (float(iou3d_correct_cnt) / float(BATCH_SIZE * 10)))
                total_correct = 0
                total_seen = 0
                loss_sum = 0
                iou2ds_sum = 0
                iou3ds_sum = 0
                iou3d_correct_cnt = 0
    def eval_one_epoch(ops):
        global EPOCH_CNT
        is_training = False
        log_string(str(datetime.now()))
        log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))
        test_idxs = np.arange(0, len(TEST_DATASET))
        num_batches = len(TEST_DATASET) / BATCH_SIZE

        # To collect statistics
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        iou2ds_sum = 0
        iou3ds_sum = 0
        iou3d_correct_cnt = 0
        # Simple evaluation with batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            #从get_batch中获取需要的tensor传入下面的feed_dict
            batch_data, batch_label, batch_center, \
            batch_hclass, batch_hres, \
            batch_sclass, batch_sres, \
            batch_rot_angle, batch_one_hot_vec = \
                get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                          NUM_POINT, NUM_CHANNEL)

            #这一块feed_dict作用是更新数据
            # 将字典ops中的参数传给占位符进行测试
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['one_hot_vec_pl']: batch_one_hot_vec,
                         ops['labels_pl']: batch_label,
                         ops['centers_pl']: batch_center,
                         ops['heading_class_label_pl']: batch_hclass,
                         ops['heading_residual_label_pl']: batch_hres,
                         ops['size_class_label_pl']: batch_sclass,
                         ops['size_residual_label_pl']: batch_sres,
                         ops['is_training_pl']: is_training}

            summary, step, loss_val, logits_val, iou2ds, iou3ds = \
                         [ops['merged'], ops['step'],
                          ops['loss'], ops['logits'],
                          ops['end_points']['iou2ds'], ops['end_points']['iou3ds']]


            #test_writer.add_summary(summary, step)

            # 降维，preds_val里存的应该是些特征向量与label对应之类的
            # np.argmax中当第二个参数axis=2时，对三维矩阵a[0][1][2]是在a[2]方向上找最大值，即在行方向比较，此时就是指在每个矩阵内部的行方向上进行比较
            preds_val = np.argmax(logits_val, 2)
            correct = np.sum(preds_val == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss_val
            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum(batch_label == l)
                total_correct_class[l] += (np.sum((preds_val == l) & (batch_label == l)))
            iou2ds_sum += np.sum(iou2ds)
            iou3ds_sum += np.sum(iou3ds)
            iou3d_correct_cnt += np.sum(iou3ds >= 0.7)

            # 蜜汁操作了一番???segp、segl看不懂 大概是看训练出来的与已有class是否匹配
            for i in range(BATCH_SIZE):
                segp = preds_val[i, :]
                segl = batch_label[i, :]
                part_ious = [0.0 for _ in range(NUM_CLASSES)]
                for l in range(NUM_CLASSES):
                    if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                        part_ious[l] = 1.0  # class not present
                    else:
                        part_ious[l] = np.sum((segl == l) & (segp == l)) / \
                                       float(np.sum((segl == l) | (segp == l)))

        log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('eval segmentation accuracy: %f' % \
                   (total_correct / float(total_seen)))
        log_string('eval segmentation avg class acc: %f' % \
                   (np.mean(np.array(total_correct_class) / \
                            np.array(total_seen_class, dtype=np.float))))
        log_string('eval box IoU (ground/3D): %f / %f' % \
                   (iou2ds_sum / float(num_batches * BATCH_SIZE), iou3ds_sum / \
                    float(num_batches * BATCH_SIZE)))
        log_string('eval box estimation accuracy (IoU=0.7): %f' % \
                   (float(iou3d_correct_cnt) / float(num_batches * BATCH_SIZE)))

        EPOCH_CNT += 1









