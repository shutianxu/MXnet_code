import sys
sys.path.insert(0, '..')

import gluonbook as gb
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import random
import zipfile




#读取歌词数据集

with zipfile.ZipFile('D:/data/jaychou_lyrics.txt.zip', 'r') as zin:
    zin.extractall('D:/data/')

with open('D:/data/jaychou_lyrics.txt', encoding='utf-8') as f:
    corpus_chars = f.read()
corpus_chars[0:50]

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')

#MINI处理
corpus_chars = corpus_chars[0:20000]

#建立词典
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
vocab_size

#建立字符索引

corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:40]
print('chars: \n', ''.join([idx_to_char[idx] for idx in sample]))
print('\nindices: \n', sample)


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))
    # 减一是因为输出的索引是相应输入的索引加一。
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y
        

def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]


ctx = gb.try_gpu()
print('will use', ctx)

num_inputs = vocab_size
num_hiddens = 256
num_outputs = vocab_size

def get_params():
    # 隐藏层参数。
    W_xh = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens),
                            ctx=ctx)
    W_hh = nd.random.normal(scale=0.01, shape=(num_hiddens, num_hiddens),
                            ctx=ctx)
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # 输出层参数。
    W_hy = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs),
                            ctx=ctx)
    b_y = nd.zeros(num_outputs, ctx=ctx)

    params = [W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params


def rnn(inputs, state, *params):
    H = state
    W_xh, W_hh, b_h, W_hy, b_y = params
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return outputs, H


def predict_rnn(rnn, prefix, num_chars, params, num_hiddens, vocab_size, ctx,
                idx_to_char, char_to_idx, get_inputs, is_lstm=False):
    prefix = prefix.lower()
    state_h = nd.zeros(shape=(1, num_hiddens), ctx=ctx)
    if is_lstm:
        # 当 RNN 使用 LSTM 时才会用到（后面章节会介绍），本节可以忽略。
        state_c = nd.zeros(shape=(1, num_hiddens), ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for i in range(num_chars + len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        # 在序列中循环迭代隐藏状态。
        if is_lstm:
            # 当 RNN 使用 LSTM 时才会用到（后面章节会介绍），本节可以忽略。
            Y, state_h, state_c = rnn(get_inputs(X, vocab_size), state_h,
                                      state_c, *params)
        else:
            Y, state_h = rnn(get_inputs(X, vocab_size), state_h, *params)
        if i < len(prefix) - 1:
            next_input = char_to_idx[prefix[i + 1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        output.append(next_input)
    return ''.join([idx_to_char[i] for i in output])

def grad_clipping(params, state_h, Y, theta, ctx):
    if theta is not None:
        norm = nd.array([0.0], ctx)
        for param in params:
            norm += (param.grad ** 2).sum()
        norm = norm.sqrt().asscalar()
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm
                
def train_and_predict_rnn(rnn, is_random_iter, num_epochs, num_steps,
                          num_hiddens, lr, clipping_theta, batch_size,
                          vocab_size, pred_period, pred_len, prefixes,
                          get_params, get_inputs, ctx, corpus_indices,
                          idx_to_char, char_to_idx, is_lstm=False):
    if is_random_iter:
        data_iter = data_iter_random
    else:
        data_iter = data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        # 如使用相邻采样，隐藏变量只需在该 epoch 开始时初始化。
        if not is_random_iter:
            state_h = nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx)
            if is_lstm:
                state_c = nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx)
        train_l_sum = nd.array([0], ctx=ctx)
        train_l_cnt = 0
        for X, Y in data_iter(corpus_indices, batch_size, num_steps, ctx):
            # 如使用随机采样，读取每个随机小批量前都需要初始化隐藏变量。
            if is_random_iter:
                state_h = nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx)
                if is_lstm:
                    state_c = nd.zeros(shape=(batch_size, num_hiddens),
                                       ctx=ctx)
            # 如使用相邻采样，需要使用 detach 函数从计算图分离隐藏状态变量。
            else:
                state_h = state_h.detach()
                if is_lstm:
                    state_c = state_c.detach()
            with autograd.record():
                # outputs 形状：(batch_size, vocab_size)。
                if is_lstm:
                    outputs, state_h, state_c = rnn(
                        get_inputs(X, vocab_size), state_h, state_c, *params)
                else:
                    outputs, state_h = rnn(
                        get_inputs(X, vocab_size), state_h, *params)
                # 设 t_ib_j 为时间步 i 批量中的元素 j：
                # y 形状：（batch_size * num_steps,）
                # y = [t_0b_0, t_0b_1, ..., t_1b_0, t_1b_1, ..., ]。
                y = Y.T.reshape((-1,))
                # 拼接 outputs，形状：(batch_size * num_steps, vocab_size)。
                outputs = nd.concat(*outputs, dim=0)
                l = loss(outputs, y)
            l.backward()
            # 裁剪梯度。
            grad_clipping(params, state_h, Y, clipping_theta, ctx)
            gb.sgd(params, lr, 1)
            train_l_sum = train_l_sum + l.sum()
            train_l_cnt += l.size
        if epoch % pred_period == 0:
            print('\nepoch %d, perplexity %f'
                  % (epoch, (train_l_sum / train_l_cnt).exp().asscalar()))
            for prefix in prefixes:
                print(' - ', predict_rnn(
                    rnn, prefix, pred_len, params, num_hiddens, vocab_size,
                    ctx, idx_to_char, char_to_idx, get_inputs, is_lstm))

num_epochs = 200
num_steps = 35
batch_size = 32
lr = 0.2
clipping_theta = 5
prefixes = ['分开', '不分开']
pred_period = 40
pred_len = 100


train_and_predict_rnn(rnn, False, num_epochs, num_steps, num_hiddens, lr,
                      clipping_theta, batch_size, vocab_size, pred_period,
                      pred_len, prefixes, get_params, get_inputs, ctx,
                      corpus_indices, idx_to_char, char_to_idx)

