
import argparse
import importlib
import os
import sys
from datetime import datetime
import numpy as np
import torch

from torch.autograd import Variable
import provider
from tensorboardX import SummaryWriter

from frustum_pointnets_v1 import FPointNet
from train_util import get_batch
from model_util import get_loss
from model_util import init_fpointnet


'''
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 2048]')
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

'''

class TrainConfig:
    def __init__(self):
        self._log_dir = "log"
        self._num_point = 1024
        self._max_epoch = 201
        self._batch_size = 32
        self._learning_rate = 0.001
        self._momentum = 0.9
        self._optimizer = 'adam'
        self._decay_step = 200000
        self._decay_rate = 0.7
        self._no_intensity = False   # 这个参数注意一下
        self._restore_model_path = None
        self._train_batch_num = None

class fpointnet_trainer :
    def __init__(self,config=TrainConfig() ):
        '''

        :param config: TrainConfig的实例
        '''
        self.config = config
        # 训练参数
        self.EPOCH_CNT = 1
        # batch_size表明这个batch中包含多少个点云数据
        self.BATCH_SIZE = config._batch_size
        # num_point表明每个点云中含有多少个点
        self.NUM_POINT = config._num_point
        self.MAX_EPOCH = config._max_epoch
        self.BASE_LEARNING_RATE = config._learning_rate
        # GPU_INDEX = config._gpu
        self.MOMENTUM = config._momentum
        self.OPTIMIZER = config._optimizer
        self.DECAY_STEP = config._decay_step
        #   初始时使用较大的学习率较快地得到较优解，随着迭代学习率呈指数逐渐减小。
        #  decayed_learning_rate = learning_rate*(decay_rate^(global_steps/decay_steps)
        self.DECAY_RATE = config._decay_rate
        self.NUM_CHANNEL = 3 if config._no_intensity else 4  # point feature channel
        self.NUM_CLASSES = 2  # segmentation has two classes
        self.LOG_DIR = config._log_dir  # 训练结果log的路径
        self.MODEL_BASE_DIR = os.path.join(self.LOG_DIR, 'models')

        if not os.path.exists(self.LOG_DIR): os.mkdir(self.LOG_DIR)
        self.LOG_FOUT = open(os.path.join(self.LOG_DIR, 'log_train.txt'), 'w')
        #self.LOG_FOUT.write(str(self.FLAGS) + '\n')

        ################################网络参数##################################

        self.BN_INIT_DECAY = 0.5
        self.BN_DECAY_DECAY_RATE = 0.5
        self.BN_DECAY_DECAY_STEP = float(self.DECAY_STEP)
        self.BN_DECAY_CLIP = 0.99

        self.TRAIN_DATASET = provider.FrustumDataset(npoints=self.NUM_POINT, split='train',
                                                rotate_to_center=True, random_flip=True, random_shift=True,
                                                one_hot=True)
        self.TEST_DATASET = provider.FrustumDataset(npoints=self.NUM_POINT, split='val',
                                               rotate_to_center=True, one_hot=True)

    def log_string(self,out_str):
        self.LOG_FOUT.write(out_str + '\n')
        self.LOG_FOUT.flush()
        print(out_str)

    def compute_summary(self,end_points, labels_pl, centers_pl, heading_class_label_pl, heading_residual_label_pl,
                        size_class_label_pl, size_residual_label_pl):
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
        for key, value in end_points.items():
            end_points_np[key] = value.cpu().data.numpy()
        iou2ds, iou3ds = provider.compute_box3d_iou(end_points_np['center'], \
                                                    end_points_np['heading_scores'],
                                                    end_points_np['heading_residuals'], \
                                                    end_points_np['size_scores'], end_points_np['size_residuals'], \
                                                    centers_pl, \
                                                    heading_class_label_pl, heading_residual_label_pl, \
                                                    size_class_label_pl, size_residual_label_pl)
        correct = torch.eq(torch.argmax(end_points['mask_logits'], dim=1),
                           labels_pl.type(torch.int64))  # end_points['mask_logits'] ,(B,2,N) , 需要调 bug
        accuracy = torch.mean(correct.type(torch.float32))
        return iou2ds, iou3ds, accuracy
    def train(self,start_index =0, train_batch_num = None, restore_modle_dir = None, train_mode = "default", input_data = None):
        '''

        :param start_index:  开始的 index
        :param train_batch_num:  训练的 batch总数
        :param restore_modle_dir:  是否载入模型
        :param train_mode:   训练模式 default : 只训练 fpointnet , all 训练整个网络， none ：不做任何训练
        :return:
        '''

        assert train_mode in ['default','all','none']
        self.start_index = start_index
        self.config._train_batch_num = train_batch_num
        self.config._restore_model_path =  restore_modle_dir
        self._train_mode = train_mode
        if self._train_mode == 'none' :
            return
        elif self._train_mode =='default' :
            self.writer = SummaryWriter('runs/default/exp')
        elif self._train_mode =='all' :
            self.writer = SummaryWriter('runs/all/exp')
        self._train();


    def _train(self, input_data = None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.config._restore_model_path == None :
            fpointnet = FPointNet()
            fpointnet = fpointnet.to(device)
            init_fpointnet(fpointnet)
        else :
            # 这个到时候可能要改成 checkpoint,现在先不管了
            fpointnet = FPointNet()
            fpointnet = fpointnet.to(device)
            fpointnet.load_state_dict(torch.load(self.config._restore_model_path))

        optimizer = torch.optim.Adam(fpointnet.parameters(), lr=self.BASE_LEARNING_RATE)

        # for name,param in fpointnet.named_parameters():
        #         print(name + ':' )
        #         print(param.requires_grad)

        for epoch in range(self.MAX_EPOCH):
            print('epoch: %d' % epoch)
            self.train_one_epoch(fpointnet, device, optimizer,input_data = None)
            self.eval_one_epoch(fpointnet, device, input_data = None)
            # save the model every 10 epoch
            if (epoch + 1) % 10 == 0:
                path = os.path.join(self.MODEL_BASE_DIR,
                                    'fpointnet_' + str(datetime.now()) + '_epoch' + str(epoch) + '.pth')
                torch.save(fpointnet.state_dict(), path)
        # save the final model
        path = os.path.join(self.MODEL_BASE_DIR, 'fpointnet_' + str(datetime.now()) + '_final' + '.pth')
        torch.save(fpointnet.state_dict(), path)

        # @torchsnooper.snoop()

    def train_one_epoch(self,fpointnet, device, optimizer,input_data = None):
        '''
        @author Qiao
        :param fpointnet: 网络
        :param device: 设备
        :return:
        '''

        self.log_string(str(datetime.now()))
        self.log_string('---- EPOCH %03d TRAINING ----' % (self.EPOCH_CNT))
        # 按照原始数据集大小进行取值
        if self.config._train_batch_num == None:
            train_idxs = np.arange(0, len(self.TRAIN_DATASET))
            np.random.shuffle(train_idxs)  # 随机
            num_batches = len(self.TRAIN_DATASET)
        else:
            num_batches = int(self.config._train_batch_num)
            num_batches = min(num_batches, len(self.TRAIN_DATASET) // self.BATCH_SIZE)
            train_idxs = np.arange(self.start_index, self.BATCH_SIZE * num_batches)
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
            start_idx = batch_idx * self.BATCH_SIZE
            end_idx = (batch_idx + 1) * self.BATCH_SIZE

            batch_data, batch_label, batch_center, \
            batch_hclass, batch_hres, \
            batch_sclass, batch_sres, \
            batch_rot_angle, batch_one_hot_vec = \
                get_batch(self.TRAIN_DATASET, train_idxs, start_idx, end_idx,
                          self.NUM_POINT, self.NUM_CHANNEL)

            pointclouds_pl = torch.from_numpy(batch_data)
            pointclouds_pl = pointclouds_pl.permute(0, 2, 1)
            pointclouds_pl = pointclouds_pl.to(device, dtype=torch.float32)
            one_hot_vec_pl = torch.from_numpy(batch_one_hot_vec)
            one_hot_vec_pl = one_hot_vec_pl.to(device, dtype=torch.float32)

            labels_pl = torch.from_numpy(batch_label).to(device, dtype=torch.int64)
            centers_pl = torch.from_numpy(batch_center).to(device, dtype=torch.float32)
            heading_class_label_pl = torch.from_numpy(batch_hclass).to(device, dtype=torch.int64)
            heading_residual_label_pl = torch.from_numpy(batch_hres).to(device, dtype=torch.float32)
            size_class_label_pl = torch.from_numpy(batch_sclass).to(device, dtype=torch.int64)
            size_residual_label_pl = torch.from_numpy(batch_sres).to(device, dtype=torch.float32)

            fpointnet.train()

            end_points = fpointnet.forward(pointclouds_pl, one_hot_vec_pl)

            if self._train_mode == "default" :
                loss, losses = get_loss(labels_pl, centers_pl, \
                                        heading_class_label_pl, heading_residual_label_pl, \
                                        size_class_label_pl, size_residual_label_pl, end_points)
            else:
                # 到时候这里要重新计算重建损失
                loss, losses = get_loss(labels_pl, centers_pl, \
                                        heading_class_label_pl, heading_residual_label_pl, \
                                        size_class_label_pl, size_residual_label_pl, end_points)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.cpu().detach().numpy()
            logits_val = end_points['mask_logits'].cpu().detach().numpy()
            iou2ds, iou3ds, accuracy = self.compute_summary(end_points, labels_pl, batch_center, \
                                                       batch_hclass, batch_hres, batch_sclass, batch_sres)
            preds_val = np.argmax(logits_val, 1)
            correct = np.sum(preds_val == batch_label)
            total_correct += correct
            total_seen += (self.BATCH_SIZE * self.NUM_POINT)
            loss_sum += loss_val
            iou2ds_sum += np.sum(iou2ds)
            iou3ds_sum += np.sum(iou3ds)
            iou3d_correct_cnt += np.sum(iou3ds >= 0.7)

            iou2d_t = np.sum(iou2ds) / float(self.BATCH_SIZE)
            iou3d_t = np.sum(iou3ds) / float(self.BATCH_SIZE)
            self.writer.add_scalar('iou2ds', iou2d_t, global_step=self.EPOCH_CNT * batch_idx)
            self.writer.add_scalar('iou3ds', iou3d_t, global_step=self.EPOCH_CNT * batch_idx)
            for key, value in losses.items():
                self.writer.add_scalar(key, losses[key].cpu().data.numpy(), global_step=self.EPOCH_CNT * batch_idx)
            # writer.add_scalar('total_loss', loss, global_step=EPOCH_CNT*batch_idx)
            for param_group in optimizer.param_groups:
                learning_rate = param_group['lr']
            self.writer.add_scalar('learning_rate', learning_rate, global_step=self.EPOCH_CNT * batch_idx)
            self.writer.add_scalar('segmentation accuracy', accuracy, global_step=self.EPOCH_CNT * batch_idx)

            if (batch_idx + 1) % 10 == 0:
                self.log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
                self.log_string('mean loss: %f' % (loss_sum / 10))
                self.log_string('segmentation accuracy: %f' % \
                           (total_correct / float(total_seen)))
                self.log_string('box IoU (ground/3D): %f / %f' % \
                           (iou2ds_sum / float(self.BATCH_SIZE * 10), iou3ds_sum / float(self.BATCH_SIZE * 10)))
                self.log_string('box estimation accuracy (IoU=0.7): %f' % \
                           (float(iou3d_correct_cnt) / float(self.BATCH_SIZE * 10)))
                total_correct = 0
                total_seen = 0
                loss_sum = 0
                iou2ds_sum = 0
                iou3ds_sum = 0
                iou3d_correct_cnt = 0
        #self.EPOCH_CNT += 1

    def eval_one_epoch(self,fpointnet, device, input_data = None):
        '''
        @author chonepieceyb, 这个函数还没有重新改
        :param fpointnet:  网络对象
        :param device: 设备
        :return:
        '''
        # get data

        self.log_string(str(datetime.now()))
        self.log_string('---- EPOCH %03d EVALUATION ----' % (self.EPOCH_CNT))
        test_idxs = np.arange(0, len(self.TEST_DATASET))
        num_batches = len(self.TEST_DATASET)

        # To collect statistics
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(self.NUM_CLASSES)]
        total_correct_class = [0 for _ in range(self.NUM_CLASSES)]
        iou2ds_sum = 0
        iou3ds_sum = 0
        iou3d_correct_cnt = 0

        fpointnet.eval()  # 训练模式
        for batch_idx in range(int(num_batches)):
            start_idx = batch_idx * self.BATCH_SIZE
            end_idx = (batch_idx + 1) * self.BATCH_SIZE
            batch_data, batch_label, batch_center, \
            batch_hclass, batch_hres, \
            batch_sclass, batch_sres, \
            batch_rot_angle, batch_one_hot_vec = \
                get_batch(self.TEST_DATASET, test_idxs, start_idx, end_idx,
                          self.NUM_POINT, self.NUM_CHANNEL)

            # convert to torch tensor and change data  format
            batch_data_gpu = torch.from_numpy(batch_data).permute(0, 2, 1).to(device, dtype=torch.float32)  #
            batch_label_gpu = torch.from_numpy(batch_label).to(device, dtype=torch.int64)
            batch_center_gpu = torch.from_numpy(batch_center).to(device, dtype=torch.float32)
            batch_hclass_gpu = torch.from_numpy(batch_hclass).to(device, dtype=torch.int64)
            batch_hres_gpu = torch.from_numpy(batch_hres).to(device, dtype=torch.float32)
            batch_sclass_gpu = torch.from_numpy(batch_sclass).to(device, dtype=torch.int64)
            batch_sres_gpu = torch.from_numpy(batch_sres).to(device, dtype=torch.float32)
            batch_one_hot_vec_gpu = torch.from_numpy(batch_one_hot_vec).to(device, dtype=torch.float32)

            # eval
            with torch.no_grad():
                end_points = fpointnet.forward(batch_data_gpu, batch_one_hot_vec_gpu)

                if self._train_mode =='default' :
                    loss, losses = get_loss(batch_label_gpu, batch_center_gpu, batch_hclass_gpu, batch_hres_gpu,
                                            batch_sclass_gpu, batch_sres_gpu, end_points)
                else :
                    loss, losses = get_loss(batch_label_gpu, batch_center_gpu, batch_hclass_gpu, batch_hres_gpu,
                                            batch_sclass_gpu, batch_sres_gpu, end_points)
            # get data   and transform dataformat from torch style to tensorflow style
            loss_val = loss.cpu().data.numpy()
            logits_val = end_points['mask_logits'].data.cpu().numpy()
            iou2ds, iou3ds, accuracy = self.compute_summary(end_points, batch_label_gpu, batch_center, batch_hclass,
                                                       batch_hres, batch_sclass, batch_sres)
            preds_val = np.argmax(logits_val, 1)
            correct = np.sum(preds_val == batch_label)
            total_correct += correct
            total_seen += (self.BATCH_SIZE * self.NUM_POINT)
            loss_sum += loss_val
            for l in range(self.NUM_CLASSES):
                total_seen_class[l] += np.sum(batch_label == l)
                total_correct_class[l] += (np.sum((preds_val == l) & (batch_label == l)))
            iou2ds_sum += np.sum(iou2ds)
            iou3ds_sum += np.sum(iou3ds)
            iou3d_correct_cnt += np.sum(iou3ds >= 0.7)

            for i in range(self.BATCH_SIZE):
                segp = preds_val[i, :]
                segl = batch_label[i, :]
                part_ious = [0.0 for _ in range(self.NUM_CLASSES)]
                for l in range(self.NUM_CLASSES):
                    if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                        part_ious[l] = 1.0  # class not present
                    else:
                        part_ious[l] = np.sum((segl == l) & (segp == l)) / \
                                       float(np.sum((segl == l) | (segp == l)))

        self.log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
        self.log_string('eval segmentation accuracy: %f' % \
                   (total_correct / float(total_seen)))
        self.log_string('eval segmentation avg class acc: %f' % \
                   (np.mean(np.array(total_correct_class) / \
                            np.array(total_seen_class, dtype=np.float))))
        self.log_string('eval box IoU (ground/3D): %f / %f' % \
                   (iou2ds_sum / float(num_batches * self.BATCH_SIZE), iou3ds_sum / \
                    float(num_batches * self.BATCH_SIZE)))
        self.log_string('eval box estimation accuracy (IoU=0.7): %f' % \
                   (float(iou3d_correct_cnt) / float(num_batches * self.BATCH_SIZE)))

        self.EPOCH_CNT += 1

if __name__ == '__main__' :
    trainer = fpointnet_trainer();
    trainer.train();