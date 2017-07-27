import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d
import time
import sys
from sklearn.metrics import accuracy_score

srng = RandomStreams()

SAVE_PATH = "/media/bento/Storage/owncloud/Biosignals/Research Projects/DeepLibphys/Current Trained/bento"

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


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


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01), borrow=True)


def init_weights_x(shape):
    # Xavier Glorot initialization
    n1, n2 = shape[0], shape[3]
    receptive_field = shape[1] * shape[2]
    std = 2.0 / np.sqrt((n1 + n2) * receptive_field)
    return theano.shared(floatX(np.random.randn(*shape) * std), borrow=True)


class CNN:
    def __init__(self):

        self.w = init_weights((64, 1, 3, 3))
        self.w2 = init_weights((128, 64, 2, 2))
        self.w3 = init_weights((256, 128, 2, 2))
        self.w4 = init_weights((4096, 500)) # 12544 60x60
        self.w_o = init_weights((500, 20))

        self.params = [self.w, self.w2, self.w3, self.w4, self.w_o]
        #self.updates = []

    def model(self, X, p_drop_conv, p_drop_hidden):
        w, w2, w3, w4, w_o = self.w, self.w2, self.w3, self.w4, self.w_o
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

    def fit(self, trX, trY, teX, teY, batch_size=5, filename=None):
        # input image shape = (batch size, input channels, input rows, input cols)
        X = T.ftensor4()
        Y = T.lmatrix()
        t_start = time.time()

        noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = self.model(X, 0.2, 0.5)

        cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))

        updates = self.RMSprop(cost, lr=0.0001, rho=0.9, epsilon=1e-6)

        train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

        if len(trX) % batch_size != 0:
            print("Length of input not a multiple of batch size")

        for i in range(len(trX) // batch_size):
            for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
                cost = train(trX[start:end], trY[start:end])
            t_end = time.time()
            print("Epoch", i, "cost:", cost, "Training time:", t_end-t_start, "Accuracy:", accuracy_score(teY,self.predict(teX)))

        if filename is not None:
            self.save(filename)
        return self

    def predict(self, teX):
        X = T.ftensor4()
        py_x = self.model(X, 0., 0.)[4]
        y_x = T.argmax(py_x, axis=1)
        predictions = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

        return predictions(teX)

    def RMSprop(self, cost, lr=0.1, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=self.params)
        updates = []
        for p, g in zip(self.params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
        return updates

    def save(self, filename, path=SAVE_PATH):
        filename = path + filename + '.npz'
        print("Saving model to file: " + filename)
        np.savez(filename,
                 w = self.w.get_value(),
                 w2 = self.w2.get_value(),
                 w3 = self.w3.get_value(),
                 w4 = self.w4.get_value(),
                 w_o = self.w_o.get_value())

    def load(self, filename, path=SAVE_PATH):
        npzfile = np.load(path + filename + ".npz")

        try:
            w = npzfile["w"]
            self.w.set_value(w)
        except:
            print("Error loading variable {0}".format("w"))
        try:
            w1 = npzfile["w1"]
            self.w1.set_value(w1)
        except:
            print("Error loading variable {0}".format("w1"))
        try:
            w2 = npzfile["w2"]
            self.w2.set_value(w2)
        except:
            print("Error loading variable {0}".format("w2"))
        try:
            w3 = npzfile["w3"]
            self.w3.set_value(w3)
        except:
            print("Error loading variable {0}".format("w3"))
        try:
            w4 = npzfile["w4"]
            self.w4.set_value(w4)
        except:
            print("Error loading variable {0}".format("w4"))
        try:
            w_o = npzfile["w_o"]
            self.w_o.set_value(w_o)
        except:
            print("Error loading variable {0}".format("w_o"))

        sys.stdout.flush()

        return self

