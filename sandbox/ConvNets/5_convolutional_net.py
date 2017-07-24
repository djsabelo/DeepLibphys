import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def init_weights_x(shape):
    # Xavier Glorot initialization
    n1, n2 = shape[0], shape[3]
    receptive_field = shape[1] * shape[2]
    std = 2.0 / np.sqrt((n1+n2)*receptive_field)
    return theano.shared(floatX(np.random.randn(*shape) * std))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def SGD(cost, params, lr=0.01, decay=1e-6, momentum=0.9, nesterov=True):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    niter = theano.shared(floatX(0.))
    lr *= 1. / (1. + decay * niter) # niter as global
    #shapes = [T.shape(p) for p in params]
    moments = [T.zeros_like(shape) for shape in shapes]
    #noise = theano.shared(np.float32(np.random.randn() * 0.001), 'noise')
    # self.w,... = niter + moments
    for p, g, m in zip(params, grads, moments):
        v = momentum * m - lr * g
        updates.append((m, v))
        if nesterov:
            new_p = p + momentum * v - lr * g
        else:
            new_p = p + v
        updates.append((p, new_p))
    niter += 1
    return updates


def Adam(cost, params, lr=0.001, beta1=0.9, beta2=.999, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    #noise = theano.shared(np.float32(np.random.randn() * 0.001), 'noise')
    updates = []
    i = theano.shared(floatX(0.))
    i_t = i + 1
    fix1 = 1. - (1. - beta1) ** i_t
    fix2 = 1. - (1. - beta2) ** i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = beta1 * g + (1-beta1) * m
        v_t = beta2 * T.sqr(g) + (1 - beta2) * v
        gradient_scaling = T.sqrt(v_t) + epsilon
        g_t = m_t / gradient_scaling
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p - lr_t * g_t))
    return updates


def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx


trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

X = T.ftensor4()
Y = T.fmatrix()

w = init_weights_x((32, 1, 3, 3))
w2 = init_weights_x((64, 32, 3, 3))
w3 = init_weights_x((128, 64, 3, 3))
w4 = init_weights((128 * 3 * 3, 625))
w_o = init_weights((625, 10))

noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
py_x = model(X, w, w2, w3, w4, 0., 0.)[4]
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print(np.mean(np.argmax(teY, axis=1) == predict(teX)))
