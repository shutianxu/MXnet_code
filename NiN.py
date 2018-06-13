# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 10:13:09 2018

@author: 1707500
"""


import sys
sys.path.insert(0, '..')
import gluonbook as gb
from mxnet import nd, gluon, init
from mxnet.gluon import nn

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size,
                      strides, padding, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk

net = nn.Sequential()
net.add(
    nin_block(96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2D(pool_size=3, strides=2),
    nin_block(256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2D(pool_size=3, strides=2),
    nin_block(384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2D(pool_size=3, strides=2), nn.Dropout(.5),
    # 标签类数是 10。
    nin_block(10, kernel_size=3, strides=1, padding=1),
    # 全局平均池化层将窗口形状自动设置成输出的高和宽。
    nn.GlobalAvgPool2D(),
    # 将四维的输出转成二维的输出，其形状为（批量大小，10）。
    nn.Flatten())

X = nd.random.uniform(shape=(1,1,224,224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
    

lr = .1
ctx = gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
train_data, test_data = gb.load_data_fashion_mnist(batch_size=128, resize=224)
gb.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=3)    
