import numpy as np

from DeepLibphys.models import LibphysSGDGRU
from DeepLibphys.utils.functions.common import segment_signal, ModelType
from DeepLibphys.utils.functions.signal2model import *
from DeepLibphys.models.LibphysGRU import LibphysGRU
import theano
import theano.tensor as T
import time
import sys
import math
from theano import printing

class LibphysMBGRU(LibphysGRU):

    def __init__(self, signal2model):
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

        super().__init__(signal2model, ModelType.MINI_BATCH, [E, U, W, V, b, c])
        self.one_matrix = theano.shared(name='one',
                                        value=np.ones((self.hidden_dim, self.mini_batch_size)).astype(theano.config.floatX))

        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        parameters = [E, V, U, W, b, c] = self.E, self.V, self.U, self.W, self.b, self.c
        x = T.imatrix('x')
        y = T.imatrix('y')
        coversion_ones = T.ones((self.mini_batch_size, 1))


        def forward_prop_step(x_t, s_prev1, s_prev2, s_prev3):
            # Embedding layer
            x_e = E[:, x_t]

            def GRU(i, U, W, b, x_0, s_previous):
                b1 = T.specify_shape((coversion_ones * b[i * 3, :]).T, T.shape(x_0))
                b2 = T.specify_shape((coversion_ones * b[i * 3 + 1, :]).T, T.shape(x_0))
                b3 = T.specify_shape((coversion_ones * b[i * 3 + 2, :]).T, T.shape(x_0))

                z = T.nnet.hard_sigmoid(U[i * 3 + 0].dot(x_0) + W[i * 3 + 0].dot(s_previous) + b1)
                r = T.nnet.hard_sigmoid(U[i * 3 + 1].dot(x_0) + W[i * 3 + 1].dot(s_previous) + b2)
                s_candidate = T.tanh(U[i * 3 + 2].dot(x_0) + W[i * 3 + 2].dot(s_previous * r) + b3)

                return (T.ones_like(z) - z) * s_candidate + z * s_previous

            # GRU Layer 1
            s1 = GRU(0, U, W, b, x_e, s_prev1)

            # GRU Layer 2
            s2 = GRU(1, U, W, b, s1, s_prev2)

            # GRU Layer 3
            s3 =GRU(2, U, W, b, s2, s_prev3)

            # Final output calculation
            c_matrix = (coversion_ones * c).T
            juju = V.dot(s3) + c_matrix

            o_t = T.nnet.softmax(juju.T).T

            return [o_t, s1, s2, s3]

        # p_o = printing.Print('prediction')
        [o, s1, s2, s3], updates = theano.scan(
            forward_prop_step,
            sequences=x.T,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=T.zeros((self.hidden_dim, self.mini_batch_size))),
                          dict(initial=T.zeros((self.hidden_dim, self.mini_batch_size))),
                          dict(initial=T.zeros((self.hidden_dim, self.mini_batch_size)))])

        def p(j, name):
            return printing.Print(name)(j)

        prediction = T.argmax(o, axis=1)

        e = ((prediction - y.T) ** 2) / (T.shape(prediction)[0] * T.shape(prediction)[1])
        cost_batch = self.calculate_ce_vector(o, y)
        mse_cost_batch = self.calculate_mean_squared_error_vecor(prediction, y)
        # Total cost
        cost = (1 / self.mini_batch_size) * self.calculate_error(o, y)

        # Gradients
        derivatives = self.calculate_gradients(cost, parameters)

        # Assign functions
        self.predict = theano.function([x], [o])
        self.predict_class = theano.function([x, y], [prediction, e], allow_input_downcast=True)
        self.error = theano.function([x, y], e)
        self.calculate_loss_vector = theano.function([x, y], cost_batch, allow_input_downcast=True)
        self.calculate_mse_vector = theano.function([x, y], mse_cost_batch, allow_input_downcast=True)
        self.ce_error = theano.function([x, y], cost, allow_input_downcast=True)
        self.bptt = theano.function([x, y], derivatives, allow_input_downcast=True)

        # SGD parameters

        # rmsprop cache updates
        self.update_RMSPROP(cost, parameters, derivatives, x, y)

    def calculate_error(self, O, Y):
        return T.sum(self.calculate_ce_vector(O, Y))

    def calculate_ce_vector(self, O, Y):
        return [T.sum(T.nnet.categorical_crossentropy(O[:, :, i], Y[i, :])) for i in range(self.mini_batch_size)]

    def calculate_mean_squared_error_vecor(self, Pred, Y):
        return T.mean(T.power(Pred.T-Y, 2), axis=1)

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(X[i:i + self.mini_batch_size, :], Y[i:i + self.mini_batch_size, :])
                       for i in range(0, np.shape(X)[0], self.mini_batch_size)])


    def calculate_loss(self, X, Y):
        num_words = float(np.shape(X)[0] * np.shape(X)[1])
        return self.calculate_total_loss(X, Y) / num_words

    def generate_predicted_signal(self, N=2000, starting_signal=[0], window_seen_by_GRU_size=256):
        signal2model = Signal2Model(self.model_name, self.get_directory_tag(), signal_dim=self.signal_dim,
                                    hidden_dim=self.hidden_dim, signal_type=self.signal_type, model_type=ModelType.SGD)

        model = LibphysSGDGRU.LibphysSGDGRU(signal2model)
        model.load(self.model_name, filetag=self.get_file_tag(-5,-5))

        return model.generate_predicted_signal(N, starting_signal, window_seen_by_GRU_size)

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