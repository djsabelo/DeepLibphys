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

class LibphysCrossGRU(LibphysGRU):

    def __init__(self, signal2model, dimensions=2):
        self.dimensions = dimensions
        self.cross_dimensions = dimensions**2

        # Assign instance variables
        E = np.random.uniform(-np.sqrt(1. / signal2model.signal_dim), np.sqrt(1. / signal2model.signal_dim),
                              (self.dimensions, signal2model.hidden_dim, signal2model.signal_dim))

        U = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                              (2*self.cross_dimensions, signal2model.hidden_dim, signal2model.hidden_dim))
        W = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                              (2*self.cross_dimensions, signal2model.hidden_dim, signal2model.hidden_dim))
        V = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                              (self.cross_dimensions, signal2model.signal_dim, signal2model.hidden_dim))

        b = np.zeros((2*self.cross_dimensions, signal2model.hidden_dim, dimensions))
        c = np.zeros((self.cross_dimensions, signal2model.signal_dim))

        super().__init__(signal2model, ModelType.MINI_BATCH, [E, U, W, V, b, c])

        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        parameters = [E, V, U, W, b, c] = self.E, self.V, self.U, self.W, self.b, self.c
        x = T.imatrix('x')
        y = T.imatrix('y')
        def p(j, name):
            return printing.Print(name)(j)

        def GRU(i, x_0, s_previous):
            z = T.nnet.sigmoid(U[i * 2 + 0].dot(x_0) + W[i * 2 + 0].dot(s_previous) + b[i * 2])
            r = T.nnet.sigmoid(U[i * 2 + 1].dot(x_0) + W[i * 2 + 1].dot(s_previous) + b[i * 2 + 1])
            s_candidate = T.tanh(U[i * 2 + 2].dot(x_0) + W[i * 2 + 2].dot(s_previous * r) + b[i * 2 + 2])

            return (T.ones_like(z) - z) * s_candidate + z * s_previous

        def forward_prop_step(x_t, s_prev):
            # Embedding layer
            x_e0 = E[0, :, x_t[0]]
            x_e1 = E[1, :, x_t[1]]
            s_ = T.zeros((4, 2, 2, 3))
            o_t = T.zeros((2, 2, 2, 5))

            # GRU BLOCK 1 [1 vs 1] #############
            # GRU Layer 1
            fi = p(x_e0, "x_e0")
            fu = GRU(0, fi, s_prev[0, 0, :])
            xx = p(fu, "fu")
            s_ = T.set_subtensor(s_[0, 0, :], xx)

            # GRU Layer 2
            s_ = T.set_subtensor(s_[0, 1], GRU(1, s_[0, 0], s_prev[0, 1]))

            # GRU BLOCK 2  [2 vs 1] #############
            # GRU Layer 1
            s_ = T.set_subtensor(s_[1, 0], GRU(2, x_e1, s_prev[1, 0]))

            # GRU Layer 2
            s_ = T.set_subtensor(s_[1, 1], GRU(3, s_[1, 0], s_prev[1, 1]))

            # GRU BLOCK 3  [1 vs 2] #############
            # GRU Layer 1
            s_ = T.set_subtensor(s_[2, 0], GRU(4, x_e0, s_prev[2, 0]))

            # GRU Layer 2
            s_ = T.set_subtensor(s_[2, 1], GRU(5, s_[2, 0], s_prev[2, 1]))

            # GRU BLOCK 4 [2 vs 2]  #############
            # GRU Layer 1
            s_ = T.set_subtensor(s_[3, 0], GRU(6, x_e1, s_prev[3, 0]))

            # GRU Layer 2
            s_ = T.set_subtensor(s_[3, 1], GRU(7, s_[3, 0], s_prev[3, 1]))

            # Final output calculation
            # FIRST DIMENSION:
            o_t = T.set_subtensor(o_t[0, 0], T.nnet.softmax((V[0].dot(s_[0, 1]) + c[1]).T).T)
            o_t = T.set_subtensor(o_t[0, 1], T.nnet.softmax((V[1].dot(s_[1, 1]) + c[2]).T).T)

            # SECOND DIMENSION:
            o_t = T.set_subtensor(o_t[1, 0], T.nnet.softmax((V[2].dot(s_[2, 1]) + c[3]).T).T)
            o_t = T.set_subtensor(o_t[1, 1], T.nnet.softmax((V[3].dot(s_[3, 1]) + c[4]).T).T)

            return [o_t, s_]

        # p_o = printing.Print('prediction')
        [o, s], updates = theano.scan(
            forward_prop_step,
            sequences=x.T,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=T.zeros((4, 2, 2, 3)))])



        prediction = T.argmax(o, axis=2)

        # e = ((prediction - np.array([[y[0], y[0]], [y[1], y[1]]])) ** 2) / (T.shape(prediction)[0] * T.shape(prediction)[1])
        cost_batch = self.calculate_ce_vector(o, y)
        # Total cost
        cost = self.calculate_error(o, y)

        # Gradients
        derivatives = self.calculate_gradients(cost, parameters)

        # Assign functions
        self.predict = theano.function([x], [o])
        # self.predict_class = theano.function([x, y], [prediction, e], allow_input_downcast=True)
        # self.error = theano.function([x, y], e)
        self.calculate_loss_vector = theano.function([x, y], cost_batch, allow_input_downcast=True)
        self.ce_error = theano.function([x, y], cost, allow_input_downcast=True)
        self.bptt = theano.function([x, y], derivatives, allow_input_downcast=True)

        # SGD parameters

        # rmsprop cache updates
        self.update_RMSPROP(cost, parameters, derivatives, x, y)

    def calculate_error(self, O, Y):
        return T.sum(self.calculate_ce_vector(O, Y))

    def calculate_ce_vector(self, O, Y):
        error = T.zeros((2, 2, 1))
        for i in range(2):
            for j in range(2):
                for zi in range(2):
                    error = T.set_subtensor(error[i, j], error[i,j]+T.nnet.categorical_crossentropy(O[i, j, zi, :], Y[j, :]))

        return error

    def calculate_total_loss(self, X, Y):
        zi = 0
        for x, y in zip(X, Y):
                zi += self.ce_error(x, y)
        return zi

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