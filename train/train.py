import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
FPOINTNET_DIR = os.path.join(ROOT_DIR,'fpointnet')
SRCNN_DIR = os.path.join(ROOT_DIR,'srcnn')
sys.path.append(SRCNN_DIR)
sys.path.append(os.path.join(FPOINTNET_DIR,'train'))

import fpointnet_trainer as FT
import srcnn_trainer as ST

def train(mode,**kwargs):
    '''
    :param mode:  训练模式，只训练 srcnn， 只训练 fpointnet 一起训练
    :param kwargs:  不同模式下所需的训练参数
    :return:
    '''
    assert mode in ['SRcnn','FPointnet','All']
    if mode == 'SRcnn' :
        strainer = ST.SrcnnTrainer();
        strainer.train();
        return
    if mode == 'FPointnet' :
        ftrainer = FT.FpontnetTrainer();
        ftrainer.train();
        return

    # 对 fconfig 进行额外的赋值

    sconfig = ST.TrainConfig();

    # 利用重建损失同时训练 srcnn 和 fpointnet

    fconfig = FT.TrainConfig()

    strainer = ST.SrcnnTrainer(sconfig)
    ftrainer = FT.FpontnetTrainer(fconfig)

    SRCNN_RESTORE_DIR = kwargs["srcnn_restore_dir"]
    FPOINTNET_RESTORE_DIR = kwargs["fpointnet_restore_dir"]
    srcnn_output = strainer.train( SRCNN_RESTORE_DIR ,"all")

    # 这里对 srcnn 的 output进行处理
    ftrainer.train(FPOINTNET_RESTORE_DIR,"all",srcnn_output)