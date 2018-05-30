import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn




def get_text_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)



def show_fashion_imgs(images):
    n = images.shape[0]
    _, figs = gb.plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    gb.plt.show()
    


X, y = mnist_train[0:9]
show_fashion_imgs(X)
print(get_text_labels(y))



net = nn.Sequential()
net.add(nn.Flatten())
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(128, activation='relu'))
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))




loss = gloss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})



num_epochs = 5
gb.train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
             None, trainer)




data, label = mnist_test[0:9]
show_fashion_imgs(data)
print('labels:', get_text_labels(label))
predicted_labels = net(data).argmax(axis=1)
print('predictions:', get_text_labels(predicted_labels.asnumpy()))