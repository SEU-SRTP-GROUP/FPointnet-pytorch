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
from model.model_util import   get_loss
from train.train_util import  get_batch
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

def compute_summary(end_points,labels_pl,centers_pl,heading_class_label_pl,heading_residual_label_pl,size_class_label_pl,size_residual_label_pl):
    '''
    计算 iou_2d, iou_3d 用 原作者提供的 numpy 版本 的操作实现可能速度会偏慢
    @author chonepeiceyb
    :param end_points:   预测结果
    :param labels_pl:      (B,2)
    :param centers_pl:      (B,3)
    :param heading_class_label_pl:   (B,)
    :param heading_residual_label_pl:(B,)
    :param size_class_label_pl:(B,)
    :param size_residual_label_pl:(B,3)
    :return:
    iou2ds: (B,) birdeye view oriented 2d box ious
    iou3ds: (B,) 3d box ious
    accuracy： python float 平均预测准确度
    '''
    end_points_np = {}
    # convert tensor to numpy array
    for key,value in end_points.item():
        end_points_np[key] = value.numpy()
    iou2ds, iou3ds = provider.compute_box3d_iou(  end_points['center'],\
                                                  end_points['heading_scores'], end_points['heading_residuals'], \
                                                  end_points['size_scores'], end_points['size_residuals'],\
                                                  centers_pl,\
                                                  heading_class_label_pl,heading_residual_label_pl,\
                                                  size_class_label_pl,size_residual_label_pl)
    correct = torch.equal( torch.argmax(end_points['mask_logits'],dim=1),labels_pl.type(torch.int64))                  #end_points['mask_logits'] ,(B,2,N) , 需要调 bug
    accuracy = torch.mean(correct.type(torch.float32))
    return iou2ds,iou3ds,accuracy

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

def eval_one_epoch(fpointnet,device):
    '''
    @author chonepieceyb
    :param fpointnet:  网络对象
    :param device: 设备
    :return:
    '''
    # get data
    global EPOCH_CNT
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET) // BATCH_SIZE

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0

    fpointnet.eval()  # 训练模式
    for batch_idx in range(int(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1)* BATCH_SIZE
        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                      NUM_POINT, NUM_CHANNEL)
        # convert to torch tensor and change data  format
        batch_data = torch.from_numpy(batch_data).permute(0,2,1).to(device)                        #
        batch_label= torch.from_numpy(batch_label).to(device)
        batch_center = torch.from_numpy(batch_center).to(device)
        batch_hclass = torch.from_numpy(batch_hclass).to(device)
        batch_hres = torch.from_numpy(batch_hres).to(device)
        batch_sclass = torch.from_numpy(batch_sclass).to(device)
        batch_sres = torch.from_numpy(batch_sres).to(device)
        batch_rot_angle = torch.from_numpy(batch_rot_angle).to(device)
        batch_one_hot_vec  = torch.from_numpy(batch_one_hot_vec ).to(device)

        # eval
        end_points = fpointnet.forward(batch_data,batch_one_hot_vec)
        '''
         get_loss(mask_label, center_label, \
             heading_class_label, heading_residual_label, \
             size_class_label, size_residual_label, \
             end_points, \
             corner_loss_weight=10.0, \
             box_loss_weight=1.0):
             
             ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'centers_pred': end_points['center'],
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}
        '''
        loss  = get_loss(batch_label,batch_center,batch_hclass,batch_hres,batch_sclass,batch_sres,end_points)
        #get data   and transform dataformat from torch style to tensorflow style
        loss_val = loss.numpy()
        logits_val = end_points['mask_logits'].numpy()
        iou2ds,iou3ds,accuracy = compute_summary(end_points,batch_label,batch_center,batch_hclass,batch_hres,batch_sclass,batch_sres)
        preds_val = np.argmax(logits_val, 1)
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

        for i in range(BATCH_SIZE):
            segp = preds_val[i,:]
            segl = batch_label[i,:]
            part_ious = [0.0 for _ in range(NUM_CLASSES)]
            for l in range(NUM_CLASSES):
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0):
                    part_ious[l] = 1.0 # class not present
                else:
                    part_ious[l] = np.sum((segl==l) & (segp==l)) / \
                        float(np.sum((segl==l) | (segp==l)))

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
if __name__ == '__main__':
    train()
