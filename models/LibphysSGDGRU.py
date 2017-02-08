import numpy as np

from DeepLibphys.utils.functions.common import segment_signal, ModelType
from DeepLibphys.utils.functions.signal2model import *
from DeepLibphys.models.LibphysGRU import LibphysGRU
import theano
import theano.tensor as T
from theano import printing

GRU_DATA_DIRECTORY = "../data/trained/"


class LibphysSGDGRU(LibphysGRU):
    def __init__(self, signal2model):


        super().__init__(signal2model, ModelType.MINI_BATCH, [E, U, W, V, b, c])
        self.mini_batch_size = 1
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c

        x = T.ivector('x')
        y = T.ivector('y')
        s = [[],[],[]]

        def forward_prop_step(x_t, s_prev):
            # Embedding layer
            x_e = E[:, x_t]

            def GRU(i, U, W, b, x_0, s_previous):
                z = T.nnet.hard_sigmoid(U[i * 3 + 0].dot(x_0) + W[i * 3 + 0].dot(s_previous) + b[i * 3])
                r = T.nnet.hard_sigmoid(U[i * 3 + 1].dot(x_0) + W[i * 3 + 1].dot(s_previous) + b[i * 3 + 1])
                s_candidate = T.tanh(U[i * 3 + 2].dot(x_0) + W[i * 3 + 2].dot(s_previous * r) + b[i * 3 + 2])

                return (T.ones_like(z) - z) * s_candidate + z * s_previous

            # GRU Layer 1
            s[0] = GRU(0, U, W, b, x_e, s_prev[0])

            # GRU Layer 2
            s[1] = GRU(1, U, W, b, s[0], s_prev[1])

            # GRU Layer 3
            s[2] = GRU(2, U, W, b, s[1], s_prev[2])

            # Final output calculation
            o_t = T.nnet.softmax(V.dot(s[2]) + c)[0]

            return [o_t, s]

        [o, s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          [dict(initial=T.zeros(self.hidden_dim)),
                           dict(initial=T.zeros(self.hidden_dim)),
                           dict(initial=T.zeros(self.hidden_dim))]])

        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        # Total cost
        cost = o_error

        # Gradients
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)

        # Assign functions
        self.predict = theano.function([x], [o], allow_input_downcast=True)
        self.predict_class = theano.function([x], prediction, allow_input_downcast=True)
        self.ce_error = theano.function([x, y], cost, allow_input_downcast=True)
        self.bptt = theano.function([x, y], [dE, dU, dW, db, dV, dc], allow_input_downcast=True)

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.In(decay, value=0.9)],
            [],
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                     ], allow_input_downcast=True)

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x, y) for x, y in zip(X, Y)])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X, Y) / float(num_words)

    def generate_predicted_signal(self, N, starting_signal, window_seen_by_GRU_size):
        # We start the sentence with the start token
        new_signal = starting_signal
        probabilities = np.zeros((self.signal_dim, N))
        state_grus = np.zeros((3, self.hidden_dim, N))
        update_gate = np.zeros((self.hidden_dim, N))
        reset_gate = np.zeros((self.hidden_dim, N))
        # Repeat until we get an end token
        print('Starting model generation')
        percent = 0
        for i in range(N):
            if int(i*100/N)% 5 == 0:
                print('.', end='')
            elif int(i*100/N)% 20 == 0:
                percent += 0.2
                print('{0}%'.format(percent))
            new_signal.append(self.generate_online_predicted_signal(starting_signal, window_seen_by_GRU_size))

        return new_signal[1:]

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
