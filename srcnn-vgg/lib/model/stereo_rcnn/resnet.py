from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.stereo_rcnn.stereo_rcnn import _StereoRCNN
from model.stereo_rcnn.vgg16 import VGG16

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
  'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
  'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  '''
  #修改FC层要3个目标的分类输出
  fc_features = model.fc.in_features
  model.fc = nn.Linear(fc_features, 9)
  '''
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class resnet(_StereoRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False):
    self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    self.vgg_path = 'data/pretrained_model/vgg16.pth'
    self.dout_base_model = 256
    self.pretrained = pretrained
    self.classes = classes

    _StereoRCNN.__init__(self, classes)

  def _init_modules(self):
    # resnet = resnet101()

    # if self.pretrained == True:
    #   print("Loading pretrained weights from %s" %(self.model_path))
    #   state_dict = torch.load(self.model_path)
    #   # state_dict = torch.load(self.vgg_path)
    #   resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})
    #
    # self.RCNN_layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    # self.RCNN_layer1 = nn.Sequential(resnet.layer1)
    # self.RCNN_layer2 = nn.Sequential(resnet.layer2)
    # self.RCNN_layer3 = nn.Sequential(resnet.layer3)
    # self.RCNN_layer4 = nn.Sequential(resnet.layer4)

    vgg = VGG16()
    state_dict = torch.load(self.vgg_path)
    # 修改状态字典中key和模型名对应
    state_dict['conv1_1.weight'] = state_dict.pop('features.0.weight')
    state_dict['conv1_1.bias'] = state_dict.pop('features.0.bias')
    state_dict['conv1_2.weight'] = state_dict.pop('features.2.weight')
    state_dict['conv1_2.bias'] = state_dict.pop('features.2.bias')
    state_dict['conv2_1.weight'] = state_dict.pop('features.5.weight')
    state_dict['conv2_1.bias'] = state_dict.pop('features.5.bias')
    state_dict['conv2_2.weight'] = state_dict.pop('features.7.weight')
    state_dict['conv2_2.bias'] = state_dict.pop('features.7.bias')
    state_dict['conv3_1.weight'] = state_dict.pop('features.10.weight')
    state_dict['conv3_1.bias'] = state_dict.pop('features.10.bias')
    state_dict['conv3_2.weight'] = state_dict.pop('features.12.weight')
    state_dict['conv3_2.bias'] = state_dict.pop('features.12.bias')
    state_dict['conv3_3.weight'] = state_dict.pop('features.14.weight')
    state_dict['conv3_3.bias'] = state_dict.pop('features.14.bias')
    state_dict['conv4_1.weight'] = state_dict.pop('features.17.weight')
    state_dict['conv4_1.bias'] = state_dict.pop('features.17.bias')
    state_dict['conv4_2.weight'] = state_dict.pop('features.19.weight')
    state_dict['conv4_2.bias'] = state_dict.pop('features.19.bias')
    state_dict['conv4_3.weight'] = state_dict.pop('features.21.weight')
    state_dict['conv4_3.bias'] = state_dict.pop('features.21.bias')
    state_dict['conv5_1.weight'] = state_dict.pop('features.24.weight')
    state_dict['conv5_1.bias'] = state_dict.pop('features.24.bias')
    state_dict['conv5_2.weight'] = state_dict.pop('features.26.weight')
    state_dict['conv5_2.bias'] = state_dict.pop('features.26.bias')
    state_dict['conv5_3.weight'] = state_dict.pop('features.28.weight')
    state_dict['conv5_3.bias'] = state_dict.pop('features.28.bias')
    state_dict['fc1.weight'] = state_dict.pop('classifier.0.weight')
    state_dict['fc1.bias'] = state_dict.pop('classifier.0.bias')
    state_dict['fc2.weight'] = state_dict.pop('classifier.3.weight')
    state_dict['fc2.bias'] = state_dict.pop('classifier.3.bias')
    state_dict['fc3.weight'] = state_dict.pop('classifier.6.weight')
    state_dict['fc3.bias'] = state_dict.pop('classifier.6.bias')
    # for k,v in state_dict.items():
    #   print(k)
    vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    self.RCNN_layer0 = nn.Sequential(vgg.conv1_1,
                                     nn.ReLU(inplace=True),
                                     vgg.conv1_2,
                                     nn.ReLU(inplace=True),
                                     vgg.maxpool1)
    self.RCNN_layer1 = nn.Sequential(vgg.conv2_1,
                                     nn.ReLU(inplace=True),
                                     vgg.conv2_2,
                                     nn.ReLU(inplace=True),
                                     vgg.maxpool2)
    self.RCNN_layer2 = nn.Sequential(vgg.conv3_1,
                                     nn.ReLU(inplace=True),
                                     vgg.conv3_2,
                                     nn.ReLU(inplace=True),
                                     vgg.conv3_3,
                                     nn.ReLU(inplace=True),
                                     vgg.maxpool3)
    self.RCNN_layer3 = nn.Sequential(vgg.conv4_1,
                                     nn.ReLU(inplace=True),
                                     vgg.conv4_2,
                                     nn.ReLU(inplace=True),
                                     vgg.conv4_3,
                                     nn.ReLU(inplace=True),
                                     vgg.maxpool4)
    self.RCNN_layer4 = nn.Sequential(vgg.conv5_1,
                                     nn.ReLU(inplace=True),
                                     vgg.conv5_2,
                                     nn.ReLU(inplace=True),
                                     vgg.conv5_3,
                                     nn.ReLU(inplace=True),
                                     vgg.maxpool5)
    '''
    ###不训练某些层
    frozen_layers = [self.RCNN_layer0, self.RCNN_layer1, self.RCNN_layer2, self.RCNN_layer3, self.RCNN_layer4]
    for layer in frozen_layers:
      for name, value in layer.named_parameters():
        value.requires_grad = False
    '''

    # Top layer
    self.RCNN_toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # reduce channel

    # Smooth layers
    self.RCNN_smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.RCNN_smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.RCNN_smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    # Lateral layers
    # 这里拼上vgg后通道数不匹配的
    self.RCNN_latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
    self.RCNN_latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
    self.RCNN_latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)


    self.RCNN_top = nn.Sequential(
      nn.Conv2d(512, 2048, kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE, padding=0),
      nn.ReLU(True),
      nn.Dropout(p=0.2),
      nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
      nn.ReLU(True),
      nn.Dropout(p=0.2)
      )
    #例子
    # self.conv = torch.nn.Sequential(
    #         OrderedDict(
    #             [
    #                 ("conv1", torch.nn.Conv2d(3, 32, 3, 1, 1)),
    #                 ("relu1", torch.nn.ReLU()),
    #                 ("pool", torch.nn.MaxPool2d(2))
    #             ]
    #         ))

    self.RCNN_kpts = nn.Sequential(
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(True),
      nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
      nn.ReLU(True)
      )

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)

    self.RCNN_bbox_pred = nn.Linear(2048, 6*self.n_classes)
    self.RCNN_dim_orien_pred = nn.Linear(2048, 5*self.n_classes)
    self.kpts_class = nn.Conv2d(256, 6, kernel_size=1, stride=1, padding=0)

    # Fix blocks
    for p in self.RCNN_layer0[0].parameters(): p.requires_grad=False
    for p in self.RCNN_layer0[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_layer3.parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_layer2.parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_layer1.parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_layer0.apply(set_bn_fix)
    self.RCNN_layer1.apply(set_bn_fix)
    self.RCNN_layer2.apply(set_bn_fix)
    self.RCNN_layer3.apply(set_bn_fix)
    self.RCNN_layer4.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_layer0.eval()
      self.RCNN_layer1.eval()
      self.RCNN_layer2.train()
      self.RCNN_layer3.train()
      self.RCNN_layer4.train()

      self.RCNN_smooth1.train()
      self.RCNN_smooth2.train()
      self.RCNN_smooth3.train()

      self.RCNN_latlayer1.train()
      self.RCNN_latlayer2.train()
      self.RCNN_latlayer3.train()

      self.RCNN_toplayer.train()
      self.RCNN_kpts.train()
      self.kpts_class.train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_layer0.apply(set_bn_eval)
      self.RCNN_layer1.apply(set_bn_eval)
      self.RCNN_layer2.apply(set_bn_eval)
      self.RCNN_layer3.apply(set_bn_eval)
      self.RCNN_layer4.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    block5 = self.RCNN_top(pool5)
    fc7 = block5.mean(3).mean(2)
    return fc7