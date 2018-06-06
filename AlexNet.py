import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import nd, init, gluon
from mxnet.gluon import nn

net = nn.Sequential()
net.add(
    # 使用较大的 11 x 11 窗口来捕获物体。同时使用步幅 4 来较大减小输出高宽。
    # 这里使用的输入通道数比 LeNet 也要大很多。
    nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 减小卷积窗口，使用填充为2来使得输入输出高宽一致。且增大输出通道数。
    nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 连续三个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，
    # 进一步增大了输出通道数。前两个卷积层后不使用池化层来减小输入的高宽。
    nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
    nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
    nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    # 使用比 LeNet 输出大数倍了全连接层。其使用丢弃层来控制复杂度。
    nn.Dense(4096, activation="relu"), nn.Dropout(.5),
    nn.Dense(4096, activation="relu"), nn.Dropout(.5),
    # 输出层。我们这里使用 FashionMNIST，所以用 10，而不是论文中的 1000。
    nn.Dense(10)
)



X = nd.random.uniform(shape=(1,1,224,224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
    
    
train_data, test_data = gb.load_data_fashion_mnist(batch_size=128, resize=224)


lr = 0.01
ctx = gb.try_gpu()
net.collect_params().initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
gb.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=5)