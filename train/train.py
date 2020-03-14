import argparse
import importlib
import os
import sys
from datetime import datetime
import numpy as np
import torch

from torch.autograd import Variable
import provider


from frustum_pointnets_v1 import FPointNet
from train_util import get_batch
from model_util import get_loss
from model_util import init_fpointnet

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
parser.add_argument('--train_batch_num', default=None, help='decide how much data to train')
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
MODEL_BASE_DIR = os.path.join(LOG_DIR,'models')

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
    for key,value in end_points.items():
        end_points_np[key] = value.cpu().data.numpy()
    iou2ds, iou3ds = provider.compute_box3d_iou(  end_points_np['center'],\
                                                  end_points_np['heading_scores'], end_points_np['heading_residuals'], \
                                                  end_points_np['size_scores'], end_points_np['size_residuals'],\
                                                  centers_pl,\
                                                  heading_class_label_pl,heading_residual_label_pl,\
                                                  size_class_label_pl,size_residual_label_pl)
    correct = torch.eq( torch.argmax(end_points['mask_logits'],dim=1),labels_pl.type(torch.int64))                  #end_points['mask_logits'] ,(B,2,N) , 需要调 bug
    accuracy = torch.mean(correct.type(torch.float32))
    return iou2ds,iou3ds,accuracy

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fpointnet = FPointNet()
    fpointnet=fpointnet.to(device)
    init_fpointnet(fpointnet)

    optimizer = torch.optim.Adam(fpointnet.parameters(), lr=BASE_LEARNING_RATE)

    for epoch in range(MAX_EPOCH):
        print('epoch: %d' % epoch)
        train_one_epoch(fpointnet, device, optimizer)
        # eval_one_epoch(fpointnet, device)
        # save the model every 10 epoch
        if (epoch+1)%10 == 0:
            path = os.path.join(MODEL_BASE_DIR,'fpointnet_'+str(datetime.now())+'_epoch'+str(epoch)+'.pth')
            torch.save(fpointnet.state_dict(),path)
    # save the final model
    path = os.path.join(MODEL_BASE_DIR, 'fpointnet_' + str(datetime.now()) + '_final' + '.pth')
    torch.save(fpointnet.state_dict(), path)
# @torchsnooper.snoop()

def train_one_epoch(fpointnet,device,optimizer):
    '''
    @author Qiao
    :param fpointnet: 网络
    :param device: 设备
    :return:
    '''
    global EPOCH_CNT

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d TRAINING ----' % (EPOCH_CNT))
    # 按照原始数据集大小进行取值
    if FLAGS.train_batch_num == None:
        train_idxs = np.arange(0, len(TRAIN_DATASET))
        np.random.shuffle(train_idxs)                                                 #随机
        num_batches = len(TRAIN_DATASET)//BATCH_SIZE
    else:
        num_batches = int(FLAGS.train_batch_num)
        num_batches = min(num_batches,len(TRAIN_DATASET)//BATCH_SIZE)
        train_idxs = np.arange(0, BATCH_SIZE*num_batches)
        np.random.shuffle(train_idxs)
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

        pointclouds_pl = torch.from_numpy(batch_data)
        pointclouds_pl = pointclouds_pl.permute(0, 2, 1)
        pointclouds_pl = pointclouds_pl.to(device,dtype=torch.float32)
        one_hot_vec_pl = torch.from_numpy(batch_one_hot_vec)
        one_hot_vec_pl = one_hot_vec_pl.to(device,dtype=torch.float32)

        labels_pl = torch.from_numpy(batch_label).to(device,dtype=torch.int64)
        centers_pl = torch.from_numpy(batch_center).to(device,dtype=torch.float32)
        heading_class_label_pl = torch.from_numpy(batch_hclass).to(device,dtype=torch.int64)
        heading_residual_label_pl = torch.from_numpy(batch_hres).to(device,dtype=torch.float32)
        size_class_label_pl = torch.from_numpy(batch_sclass).to(device,dtype=torch.int64)
        size_residual_label_pl = torch.from_numpy(batch_sres).to(device,dtype=torch.float32)

        fpointnet.train()

        fpointnet.zero_grad()

        end_points = fpointnet.forward(pointclouds_pl, one_hot_vec_pl)
        loss,_ = get_loss(labels_pl, centers_pl,\
                  heading_class_label_pl, heading_residual_label_pl,\
                  size_class_label_pl, size_residual_label_pl, end_points)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.cpu().detach().numpy()
        logits_val = end_points['mask_logits'].cpu().detach().numpy()
        iou2ds,iou3ds,accuracy = compute_summary(end_points,labels_pl ,batch_center,\
                                                 batch_hclass,batch_hres,batch_sclass,batch_sres)
        preds_val = np.argmax(logits_val, 1)
        correct = np.sum(preds_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        iou3d_correct_cnt += np.sum(iou3ds>=0.7)
        if (batch_idx+1)%10 == 0:
                log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
                log_string('mean loss: %f' % (loss_sum / 10))
                log_string('segmentation accuracy: %f' % \
                    (total_correct / float(total_seen)))
                log_string('box IoU (ground/3D): %f / %f' % \
                    (iou2ds_sum / float(BATCH_SIZE*10), iou3ds_sum / float(BATCH_SIZE*10)))
                log_string('box estimation accuracy (IoU=0.7): %f' % \
                    (float(iou3d_correct_cnt)/float(BATCH_SIZE*10)))
                total_correct = 0
                total_seen = 0
                loss_sum = 0
                iou2ds_sum = 0
                iou3ds_sum = 0
                iou3d_correct_cnt = 0
    EPOCH_CNT += 1

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
        batch_data_gpu = torch.from_numpy(batch_data).permute(0,2,1).to(device,dtype=torch.float32)                        #
        batch_label_gpu= torch.from_numpy(batch_label).to(device,dtype=torch.int64)
        batch_center_gpu = torch.from_numpy(batch_center).to(device,dtype=torch.float32)
        batch_hclass_gpu = torch.from_numpy(batch_hclass).to(device,dtype=torch.int64)
        batch_hres_gpu = torch.from_numpy(batch_hres).to(device,dtype=torch.float32)
        batch_sclass_gpu = torch.from_numpy(batch_sclass).to(device,dtype=torch.int64)
        batch_sres_gpu = torch.from_numpy(batch_sres).to(device,dtype=torch.float32)
        batch_one_hot_vec_gpu  = torch.from_numpy(batch_one_hot_vec).to(device ,dtype=torch.float32)

        # eval
        with torch.no_grad():
            end_points = fpointnet.forward(batch_data_gpu,batch_one_hot_vec_gpu)
            loss  = get_loss(batch_label_gpu,batch_center_gpu,batch_hclass_gpu,batch_hres_gpu,batch_sclass_gpu,batch_sres_gpu,end_points)
        #get data   and transform dataformat from torch style to tensorflow style
        loss_val = loss.cpu().data.numpy()
        logits_val = end_points['mask_logits'].data.cpu().numpy()
        iou2ds,iou3ds,accuracy = compute_summary(end_points,batch_label_gpu,batch_center,batch_hclass,batch_hres,batch_sclass,batch_sres)
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
