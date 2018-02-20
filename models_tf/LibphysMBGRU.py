import numpy as np

from DeepLibphys.models_tf import LibphysSGDGRU
from DeepLibphys.utils.functions.signal2model import *
from DeepLibphys.models_tf.LibphysGRU import LibphysGRU
import tensorflow as tf
from scipy.stats import norm
import time
import sys
import math
from theano import printing


class LibphysMBGRU(LibphysGRU):

    def __init__(self, signal2model=None):
        if signal2model is not None:
            # Assign instance variables
            E = np.random.uniform(-np.sqrt(1. / signal2model.signal_dim), np.sqrt(1. / signal2model.signal_dim),
                                  (signal2model.hidden_dim, signal2model.signal_dim))

            U = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                                  (9, signal2model.hidden_dim, signal2model.hidden_dim))
            W = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                                  (9, signal2model.hidden_dim, signal2model.hidden_dim))
            V = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                                  (signal2model.signal_dim, signal2model.hidden_dim))

            b = np.zeros((9, signal2model.hidden_dim))
            c = np.zeros(signal2model.signal_dim)
            self.x = tf.Variable('x')
            self.y = tf.Variable('y')

            super().__init__(signal2model, ModelType.MINI_BATCH, [E, U, W, V, b, c])

        else:
            super().__init__(None, ModelType.MINI_BATCH, None)

    def forward_prop_step(self, x_t, s_prev1, s_prev2, s_prev3):
        [E, V, U, W, b, c] = self.E, self.V, self.U, self.W, self.b, self.c
        coversion_ones = tf.ones((self.mini_batch_size, 1))
        # Embedding layer
        x_e = E[:, x_t]

        def GRU(i, x_0, s_prev):
            b1 = tf.matmul(coversion_ones, b[i * 3, :]).T
            b2 = tf.matmul(coversion_ones, b[i * 3 + 1, :]).T
            b3 = tf.matmul(coversion_ones, b[i * 3 + 2, :]).T

            z = tf.nn.sigmoid(tf.matmul(U[i * 3 + 0], x_0) + tf.matmul(W[i * 3 + 0], s_prev) + b1)
            r = tf.nn.sigmoid(tf.matmul(U[i * 3 + 1], x_0) + tf.matmul(W[i * 3 + 1], s_prev) + b2)
            value = tf.matmul(U[i * 3 + 2], x_0) + tf.matmul(W[i * 3 + 2], tf.matmul(s_prev, r)) + b3
            s_candidate = tf.nn.tanh(value)

            return tf.matmul((tf.ones_like(z) - z), s_candidate) + tf.matmul(z, s_prev)

        # GRU Layer 1
        s1 = GRU(0, U, W, b, x_e, s_prev1)

        # GRU Layer 2
        s2 = GRU(1, U, W, b, s1, s_prev2)

        # GRU Layer 3
        s3 = GRU(2, U, W, b, s2, s_prev3)

        # Final output calculation
        c_matrix = (coversion_ones * c).T
        linear_node = V.dot(s3) + c_matrix

        o_t = tf.nn.softmax(linear_node.T).T

        return [o_t, s1, s2, s3]


        # p_o = printing.Print('prediction')

    def propagate(self, x):
        initializer = [None,
                       dict(initial=tf.zeros((self.hidden_dim, self.mini_batch_size))),
                       dict(initial=tf.zeros((self.hidden_dim, self.mini_batch_size))),
                       dict(initial=tf.zeros((self.hidden_dim, self.mini_batch_size)))]
        [o, s1, s2, s3] = tf.scan(self.forward_prop_step, x.T, initializer=initializer, parallel_iterations=0)
        return [o, s1, s2, s3]

    def predict(self, x):
        [o, s1, s2, s3] = self.propagate(x)
        return tf.argmax(o, axis=1)

    def loss(self, x, y):
        o = self.predict(x)
        loss, backprop = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=o)
        return loss

    def cost_batch(self):


    cost_batch = tf.square(self.prediction - y)
    self.cost = tf.reduce_mean(tf.square(self.prediction - y))
    # update RMSProp
    self.train_object = self.update_RMSPROP(self.loss)

    def predict(self, X, Y):
        [o, s1, s2, s3] = tf.scan(forward_prop_step, x.T, initializer=initializer, parallel_iterations=0)
        return tf.argmax(o, axis=1)



    def calculate_error(self, O, Y):
        return

    def calculate_ce_vector(self, O, Y):
        return [T.sum(T.nnet.categorical_crossentropy(O[:, :, i], Y[i, :])) for i in range(self.mini_batch_size)]

    def calculate_mean_squared_error_vector(self, Pred, Y):
        return T.mean(T.power(Pred.T-Y, 2), axis=1)

    def calculate_total_loss(self, X, Y):
        num_words = float(np.shape(X)[0] * np.shape(X)[1])
        return np.sum([self.ce_error(X[i:i + self.mini_batch_size, :], Y[i:i + self.mini_batch_size, :])
                       for i in range(0, np.shape(X)[0], self.mini_batch_size)]) / num_words

    def calculate_loss(self, X, Y):
        return self.calculate_total_loss(X, Y)

    def generate_predicted_signal(self, signal2model, N=2000, starting_signal=[0], window_seen_by_GRU_size=256):
        signal2model.model_type = ModelType.SGD
        model = LibphysSGDGRU.LibphysSGDGRU(signal2model)
        model.load(dir_name=signal2model.signal_directory, file_tag=self.get_file_tag(-5,-5))

        return model.generate_predicted_signal(N, starting_signal, window_seen_by_GRU_size)

    def generate_predicted_vector(self, N=2000, starting_signal=[0], window_seen_by_GRU_size=256):
        # We start the sentence with the start token
        new_signal = np.array(starting_signal).reshape((len(starting_signal, 1)))
        # Repeat until we get an end token
        print('Starting model generation')
        percent = 0
        for i in range(N):
            if int(i * 100 / N) % 5 == 0:
                print('.', end='')
            elif int(i * 100 / N) % 20 == 0:
                percent += 0.2
                print('{0}%'.format(percent))
            new_signal.append(self.generate_online_predicted_vector(starting_signal, window_seen_by_GRU_size))

        return new_signal[1:]

        return model.generate_predicted_signal(N, starting_signal, window_seen_by_GRU_size)

    def generate_online_predicted_vector(self, starting_signal, window_seen_by_GRU_size, uncertaintly=0.05):
        new_signal = np.array(starting_signal)
        next_sample = None
        xxx = None
        next_sample_probs = [12]
        # try:
        if len(new_signal) <= window_seen_by_GRU_size:
            signal = new_signal
        else:
            signal = new_signal[-window_seen_by_GRU_size:]

        [output] = self.predict(signal)
        next_sample_probs = np.asarray(output, dtype=float)
        next_sample_probs = next_sample_probs[-1] / np.sum(next_sample_probs[-1], axis=0)

        means = np.where(next_sample_probs[-1] / np.sum(next_sample_probs[-1], axis=0) > 0.01)[0]
        xxx = np.zeros(64)
        # for i in range(np.shape(means)[-1]):
        #     xxx += norm.pdf(np.arange(64), means[:, i], uncertaintly) * next_sample_probs[means[:, i]]

        xxx = xxx / np.sum(xxx, axis=0)

        # next_sample_probs = np.random.multinomial(1, next_sample_probs)
        next_sample = np.random.choice(64, p=xxx, size=(np.shape(xxx)[0]))
        # next_sample = np.argmax(sample)
        # except:
        #     print("exception: " + np.sum(np.asarray(next_sample_probs[-1]), dtype=float))
        #     next_sample = 0
        # plt.plot(np.arange(len(next_sample_probs)), next_sample_probs, 'k-', next_sample, xxx[next_sample], 'ro')
        # plt.plot(np.arange(len(xxx)), xxx, 'g-')
        # print(next_sample)
        # plt.show()
        return next_sample

    def _get_new_parameters(self):
        E = np.random.uniform(-np.sqrt(1. / self.signal_dim), np.sqrt(1. / self.signal_dim),
                              (self.hidden_dim, self.signal_dim))

        U = np.random.uniform(-np.sqrt(1. / self.hidden_dim), np.sqrt(1. / self.hidden_dim),
                              (9, self.hidden_dim, self.hidden_dim))
        W = np.random.uniform(-np.sqrt(1. / self.hidden_dim), np.sqrt(1. / self.hidden_dim),
                              (9, self.hidden_dim, self.hidden_dim))
        V = np.random.uniform(-np.sqrt(1. / self.hidden_dim), np.sqrt(1. / self.hidden_dim),
                              (self.signal_dim, self.hidden_dim))

        b = np.zeros((9, self.hidden_dim))
        c = np.zeros(self.signal_dim)

        return [E, U, W, V, b, c]