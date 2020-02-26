import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):
    def __init__(self):
        '''
        说明： 配置类，保存着 FPointNet所需要的所有参数，如果需要参数就在Config类的构造函数里面加,参数用大写加下划线表示 eg: IS_TRAINING, BN_DECAY
        '''

class FPointNet(nn.Module):
    def __init__(self,config):
        '''
        说明： 1 把需要训练的层在这里定义， 层的命名规则 层名_子模块缩写_编号 eg: self.conv_box3d_1
              2  end_points 保存着每一个模块的计算结果，是一个字典，在每一个模块结束之后对 end_points的内容进行简要说明
        '''
        super(FPointNet,self).__init__()
        self.config = config
        self.end_points = {}             #所有输出构成的字典
