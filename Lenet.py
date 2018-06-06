'''
在“多层感知机——从零开始”一节里我们构造了一个两层感知机模型来对FashionMNIST里图片进行分类。每张图片高宽均是28，我们将其展开成长为784的向量输入到模型里。这样的做法虽然简单，但也有局限性：

垂直方向接近的像素在这个向量的图片表示里可能相距很远，它们组成的模式难被模型识别。
对于大尺寸的输入图片，我们会得到过大的模型。假设输入是高宽为1000的彩色照片，即使隐藏层输出仍是256，这一层的模型形状是3,000,000×256，其占用将近3GB的内存，这带来过复杂的模型和过高的存储开销。
卷积层尝试解决这两个问题：它保留输入形状，使得可以有效的发掘水平和垂直两个方向上的数据关联。它通过滑动窗口将卷积核重复作用在输入上，而得到更紧凑的模型参数表示。

卷积神经网络就是主要由卷积层组成的网络，本小节里我们将介绍一个早期用来识别手写数字图片的卷积神经网络：LeNet [1]，其名字来源于论文一作Yann LeCun。LeNet证明了通过梯度下降训练卷积神经网络可以达到手写数字识别的最先进的结果。这个奠基性的工作第一次将卷积神经网络推上舞台，为世人所知。
'''
import sys
sys.path.append('..')
import gluonbook as gb
import mxnet as mx
from mxnet import nd, gluon, init
from mxnet.gluon import nn

net = nn.Sequential()
net.add(
    nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    # Dense 会默认将（批量大小，通道，高，宽）形状的输入转换成
    #（批量大小，通道 x 高 x 宽）形状的输入。
    nn.Dense(120, activation='sigmoid'),
    nn.Dense(84, activation='sigmoid'),
    nn.Dense(10)
)



X = nd.random.uniform(shape=(1,1,28,28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
    
    
train_data, test_data = gb.load_data_fashion_mnist(batch_size=256)


try:
    ctx = mx.gpu()
    _ = nd.zeros((1,), ctx=ctx)
except:
    ctx = mx.cpu()



ctx
lr = 1
net.collect_params().initialize(ctx=ctx, init=init.Xavier(),force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
gb.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=5)


