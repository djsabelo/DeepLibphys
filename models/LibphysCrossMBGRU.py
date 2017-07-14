import numpy as np

from DeepLibphys.models import LibphysSGDGRU
from DeepLibphys.utils.functions.common import segment_signal, ModelType
from DeepLibphys.utils.functions.signal2model import *
from DeepLibphys.models.LibphysGRU import LibphysGRU
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import time
import sys
import math
from theano import printing

class LibphysCrossMBGRU(LibphysGRU):

    def __init__(self, signal2model, dimensions=2):
        self.dimensions = dimensions
        self.cross_dimensions = dimensions**2

        # Assign instance variables
        E = np.random.uniform(-np.sqrt(1. / signal2model.signal_dim), np.sqrt(1. / signal2model.signal_dim),
                              (self.dimensions, signal2model.hidden_dim, signal2model.signal_dim))

        U = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                              (2*3*self.cross_dimensions, signal2model.hidden_dim, signal2model.hidden_dim))
        W = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                              (2*3*self.cross_dimensions, signal2model.hidden_dim, signal2model.hidden_dim))
        V = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                              (self.cross_dimensions, signal2model.signal_dim, signal2model.hidden_dim))

        b = np.zeros((3*2*self.cross_dimensions, signal2model.hidden_dim))
        c = np.zeros((self.cross_dimensions, signal2model.signal_dim))


        super().__init__(signal2model, ModelType.CROSS_MBSGD, [E, U, W, V, b, c])

        # signal = np.zeros((6, 2, 11))
        # for i in range(np.shape(signal)[0]):
        #     signal[i, 0, :] = np.array(np.random.randint(0, 4, 11), dtype=np.int32)
        #     signal[i, 1, :] = np.array((2 + np.sin(np.arange(0, 11) * np.pi) * 2 / 6), dtype=np.int32)
        #
        # self.x = np.array(x_train, dtype=np.int32)
        # self.y = np.array(y_train, dtype=np.int32)
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        parameters = [E, V, U, W, b, c] = self.E, self.V, self.U, self.W, self.b, self.c
        x = T.itensor3('x')
        y = T.itensor3('y')
        conversion_ones = T.specify_shape(T.ones((self.mini_batch_size, 1)), T.shape((self.mini_batch_size, 1)))

        # X_ = T.imatrix('X')
        # Y_ = T.imatrix('Y')

        # x.tag.test_value = self.x
        # y.tag.test_value = self.y
        # print(x.T.tag.test_value)
        cross, dim, s_d, h_d, mb = \
            self.cross_dimensions, self.dimensions, self.signal_dim, self.hidden_dim, self.mini_batch_size
        def p(j, name):
            return printing.Print(name)(j)

        def GRU(i, x_0, s_previous):
            z = T.nnet.sigmoid(x_0.dot(U[i * 3 + 0]) + s_previous.dot(W[i * 3 + 0]) + conversion_ones.dot(b[i * 3]))
            r = T.nnet.sigmoid(U[i * 3 + 1].dot(x_0) + W[i * 3 + 1].dot(s_previous) + conversion_ones.dot(b[i * 3 + 1]))
            s_candidate = T.tanh(U[i * 3 + 2].dot(x_0) +
                                 W[i * 3 + 2].dot(s_previous * r) +
                                 conversion_ones.dot(b[i * 3 + 2]))

            return (T.ones_like(z) - z) * s_candidate + z * s_previous

        def forward_prop_step(x_t, s_prev):
            # Embedding layer
            x_e0 = E[0, :, x_t[:, 0]]
            x_e1 = E[1, :, x_t[:, 1]]
            s_ = T.zeros_like(s_prev)

            # GRU BLOCK 1 [1 vs 1] #############
            # GRU Layer 1
            # print(x_e0)
            # print(s_prev)
            s_ = T.set_subtensor(s_[:, :, 0], T.stack(
                [GRU(0, x_e0, s_prev[:, 0, 0]),            # GRU BLOCK 1 [1 vs 1] #############
                 GRU(2, x_e1, s_prev[:, 1, 0]),            # GRU BLOCK 2 [2 vs 1] #############
                 GRU(4, x_e0, s_prev[:, 2, 0]),            # GRU BLOCK 3 [1 vs 2]  #############
                 GRU(6, x_e1, s_prev[:, 3, 0]),            # GRU BLOCK 4 [2 vs 2]  #############
                 ]))

            s_ = T.set_subtensor(s_[:, :, 1], T.stack(
                [GRU(1, s_[:, 0, 0], s_prev[:, 0, 1]),        # GRU BLOCK 1 [1 vs 1] #############
                 GRU(3, s_[:, 1, 0], s_prev[:, 1, 1]),        # GRU BLOCK 2 [2 vs 1] #############
                 GRU(5, s_[:, 2, 0], s_prev[:, 2, 1]),        # GRU BLOCK 3 [1 vs 2]  #############
                 GRU(7, s_[:, 3, 0], s_prev[:, 3, 1]),        # GRU BLOCK 4 [2 vs 2]  #############
                 ]))

            # Final output calculation
            # FIRST DIMENSION:
            o_t = T.stack(T.nnet.softmax((V[0].dot(s_[:, 0, 1]) + conversion_ones.dot(c[0])))[0],
                          T.nnet.softmax((V[1].dot(s_[:, 1, 1]) + conversion_ones.dot(c[1])))[0],
                          T.nnet.softmax((V[2].dot(s_[:, 2, 1]) + conversion_ones.dot(c[2])))[0],
                          T.nnet.softmax((V[3].dot(s_[:, 3, 1]) + conversion_ones.dot(c[3])))[0])

            return [o_t, s_]

        # p_o = printing.Print('prediction')
        [o, s], updates = theano.scan(
            forward_prop_step,
            sequences=x.T,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=T.zeros((self.mini_batch_size,
                                                self.cross_dimensions,
                                                self.dimensions,
                                                self.hidden_dim)))])

        cost_batch = self.calculate_ce_vector(o, y)
        # Total cost
        cost = self.calculate_error(o, y)

        # Gradients
        derivatives = self.calculate_gradients(cost, parameters)

        # Assign functions
        # self.predict = theano.function([x], [o])
        # self.predict_class = theano.function([x, y], [prediction, e], allow_input_downcast=True)
        # self.error = theano.function([x, y], e)

        self.calculate_loss_vector = theano.function([x, y], cost_batch, allow_input_downcast=True)
        self.ce_error = theano.function([x, y], cost, allow_input_downcast=True)
        self.bptt = theano.function([x, y], derivatives, allow_input_downcast=True)

        # SGD parameters

        # rmsprop cache updates
        self.update_RMSPROP(cost, parameters, derivatives, x, y)

    def p(self, j, name):
        return printing.Print(name)(j)

    def calculate_error(self, O, Y):
        return T.sum(self.calculate_ce_vector(O, Y))

    def calculate_ce_vector(self, O, Y):
        return [T.nnet.categorical_crossentropy(O[:, j], Y[int(j/self.dimensions)]) for j in range(self.cross_dimensions)]

    def calculate_ce_vector(self, O, Y):
        return [T.sum(T.nnet.categorical_crossentropy(O[:, :, i], Y[int(i/self.dimensions), :])) for i in range(self.mini_batch_size)]

    def calculate_total_loss(self, X, Y):
        return np.sum([self.calculate_loss_vector(xi, yi) for xi, yi in zip(X, Y)])

    def calculate_loss(self, X, Y):
        num_words = np.shape(X)[0] * np.shape(X)[1]
        return self.calculate_total_loss(X, Y) / float(num_words)


    def generate_predicted_signal(self, N=2000, starting_signal=[0], window_seen_by_GRU_size=256):
        # We start the sentence with the start token
        new_signal = starting_signal
        # Repeat until we get an end token
        print('Starting model generation')
        percent = 0
        for i in range(N):
            if int(i * 100 / N) % 5 == 0:
                print('.', end='')
            elif int(i * 100 / N) % 20 == 0:
                percent += 0.2
                print('{0}%'.format(percent))
            new_signal.append(self.generate_online_predicted_signal(starting_signal, window_seen_by_GRU_size))

        return new_signal[1:]


    #TODO: this is buggy for sure, correct it!!
    def generate_online_predicted_signal(self, starting_signal, window_seen_by_GRU_size):
        new_signal = starting_signal
        next_sample = None
        try:
            if len(new_signal) <= window_seen_by_GRU_size:
                signal = new_signal
            else:
                signal = new_signal[-window_seen_by_GRU_size:]

            [output] = self.predict(signal)
            next_sample_probs = np.asarray(output, dtype=float)
            sample = np.random.multinomial(1, next_sample_probs[-1] / np.sum(
                next_sample_probs[-1]))
            next_sample = np.argmax(sample)
        except:
            print("exception: " + np.sum(np.asarray(next_sample_probs[-1]), dtype=float))
            next_sample = 0

        return next_sample