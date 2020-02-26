'''
D:\srtp\frustum-pointnets-master\train\tensorboard.py
'''
import tensorflow as tf
import os
from tensorflow.python.platform import gfile

#这是从文件格式的meta文件加载模型
graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
# graphdef.ParseFromString(gfile.FastGFile("/data/TensorFlowAndroidMNIST/app/src/main/expert-graph.pb", "rb").read())
# _ = tf.import_graph_def(graphdef, name="")
_ = tf.train.import_meta_graph("./model.ckpt.meta")
summary_write = tf.summary.FileWriter("./" , graph)

'''
然后再启动tensorboard：

tensorboard --logdir /data/TensorFlowAndroidMNIST/logdir --host 你的ip --port 你端口(默认6006)
'''
