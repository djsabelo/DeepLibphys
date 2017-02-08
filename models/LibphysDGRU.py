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

class LibphysDGRU(LibphysGRU):

    def __init__(self, signal2model, dimensions):
        self.dimensions = dimensions

        # Assign instance variables
        E = np.random.uniform(-np.sqrt(1. / signal2model.signal_dim), np.sqrt(1. / signal2model.signal_dim),
                              (signal2model.hidden_dim, signal2model.signal_dim))

        U = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                              (9, signal2model.hidden_dim, signal2model.hidden_dim))
        W = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                              (9, signal2model.hidden_dim, signal2model.hidden_dim))
        V = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                              (signal2model.signal_dim, signal2model.hidden_dim))

        b = np.zeros((9, signal2model.hidden_dim, dimensions))
        c = np.zeros(signal2model.signal_dim, dimensions)

        super().__init__(signal2model, ModelType.MINI_BATCH, [E, U, W, V, b, c])

        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        parameters = [E, V, U, W, b, c] = self.E, self.V, self.U, self.W, self.b, self.c
        x = T.imatrix('x')
        y = T.imatrix('y')


        def GRU(i, U, W, b, x_0, s_previous):
            z = T.nnet.hard_sigmoid(U[i * 3 + 0].dot(x_0) + W[i * 3 + 0].dot(s_previous) + b[i * 3])
            r = T.nnet.hard_sigmoid(U[i * 3 + 1].dot(x_0) + W[i * 3 + 1].dot(s_previous) + b[i * 3 + 1])
            s_candidate = T.tanh(U[i * 3 + 2].dot(x_0) + W[i * 3 + 2].dot(s_previous * r) + b[i * 3 + 2])

            return (T.ones_like(z) - z) * s_candidate + z * s_previous

        def forward_prop_step(x_t, s_prev1, s_prev2, s_prev3):
            # Embedding layer
            x_e = E[:, x_t]

            # GRU Layer 1
            s1 = GRU(0, U, W, b, x_e, s_prev1)

            # GRU Layer 2
            s2 = GRU(1, U, W, b, s1, s_prev2)

            # GRU Layer 3
            s3 =GRU(2, U, W, b, s2, s_prev3)

            # Final output calculation
            juju = V.dot(s3) + c

            o_t = T.nnet.softmax(juju.T).T

            return [o_t, s1, s2, s3]

        # p_o = printing.Print('prediction')
        [o, s1, s2, s3], updates = theano.scan(
            forward_prop_step,
            sequences=x.T,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=T.zeros((self.hidden_dim, self.dimensions))),
                          dict(initial=T.zeros((self.hidden_dim, self.dimensions))),
                          dict(initial=T.zeros((self.hidden_dim, self.dimensions)))])

        def p(j, name):
            return printing.Print(name)(j)

        prediction = T.argmax(o, axis=1)

        e = ((prediction - y.T) ** 2) / (T.shape(prediction)[0] * T.shape(prediction)[1])
        cost_batch = self.calculate_ce_vector(o, y)
        # Total cost
        cost = self.calculate_error(o, y)

        # Gradients
        derivatives = self.calculate_gradients(cost, parameters)

        # Assign functions
        self.predict = theano.function([x], [o])
        self.predict_class = theano.function([x, y], [prediction, e], allow_input_downcast=True)
        self.error = theano.function([x, y], e)
        self.calculate_loss_vector = theano.function([x, y], cost_batch, allow_input_downcast=True)
        self.ce_error = theano.function([x, y], cost, allow_input_downcast=True)
        self.bptt = theano.function([x, y], derivatives, allow_input_downcast=True)

        # SGD parameters

        # rmsprop cache updates
        self.update_RMSPROP(cost, parameters, derivatives, x, y)

    def calculate_error(self, O, Y):
        return T.sum(self.calculate_ce_vector(O, Y))

    def calculate_ce_vector(self, O, Y):
        return [T.sum(T.nnet.categorical_crossentropy(O[:, :, i], Y[i, :])) for i in range(self.dimensions)]

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(X[i:i + self.dimensions, :], Y[i:i + self.dimensions, :])
                       for i in range(0, np.shape(X)[0], self.dimensions)])

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