import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import theano as theano
import theano.tensor as T
from theano import printing

from BiosignalsDeepLibphys.utils.functions.common import segment_signal

GRU_DATA_DIRECTORY = "../data/trained/"


# clean
# test
# correct loss correction
# include a sythesis model with scan
# create shared variables
# put on validation with test
# include the classification mode



class LibPhys_RGRU:
    def __init__(self, signal_name=None, hidden_dim=256):
        # Assign instance variables
        self.hidden_dim = hidden_dim
        self.bptt_truncate = -1
        self.current_learning_rate = 0.01
        self.signal_name = signal_name
        self.signal_dim = "RG"
        self.start_time = 0

        # Initialize the network parameters
        E = np.random.uniform(-np.sqrt(1. / self.hidden_dim), np.sqrt(1. / self.hidden_dim), (hidden_dim))
        U = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (9, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (9, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim))
        b = np.zeros((9, hidden_dim))
        c = np.zeros(1)
        v = np.zeros(hidden_dim)

        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        self.v = theano.shared(name='v', value=v.astype(theano.config.floatX))

        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        self.mv = theano.shared(name='mv', value=np.zeros(v.shape).astype(theano.config.floatX))

        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        E, V, U, W, b, c, v = self.E, self.V, self.U, self.W, self.b, self.c, self.v
        x = T.dvector('x')
        y = T.dvector('y')

        def forward_prop_step(x_t, s_t1_prev, s_t2_prev, s_t3_prev):
            p_o = printing.Print('juju')
            # Word embedding layer
            x_e = x_t.dot(E).T + v

            def GRU(i, U, W, b, x_0, s_prev):
                b1 = b[i * 3, :]
                b2 = b[i * 3 + 1, :]
                b3 = b[i * 3 + 2, :]

                z = T.nnet.hard_sigmoid(U[i * 3 + 0].dot(x_0) + W[i * 3 + 0].dot(s_prev) + b1)
                r = T.nnet.hard_sigmoid(U[i * 3 + 1].dot(x_0) + W[i * 3 + 1].dot(s_prev) + b2)
                c = T.tanh(U[i * 3 + 2].dot(x_0) + W[i * 3 + 2].dot(s_prev * r) + b3)

                return ((T.ones_like(z) - z) * c + z * s_prev).astype(theano.config.floatX)

            p_o = printing.Print('juju')
            s = [[], [], []]
            # GRU Layer 1
            s[0] = GRU(0, U, W, b, x_e, s_t1_prev)

            # GRU Layer 2
            s[1] = GRU(1, U, W, b, s[0], s_t2_prev)

            # GRU Layer 3
            s[2] = GRU(2, U, W, b, s[1], s_t3_prev)

            # Final output calculation

            o_t = (V.dot(s[2]) + c)[0]

            return [o_t, s[0], s[1], s[2]]

        # p_o = printing.Print('prediction')
        [o, s, s2, s3], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=T.zeros(self.hidden_dim)),
                          dict(initial=T.zeros(self.hidden_dim)),
                          dict(initial=T.zeros(self.hidden_dim))])

        p_o = printing.Print('o')
        # p_y = printing.Print('y')
        prediction = o
        e = prediction - y
        o_last = o[-1]
        o_error = T.sum(T.pow(prediction.T - y, 2)) / (2 * T.shape(y)[1])
        # Total cost (could add regularization here)
        cost = o_error

        # Gradients
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)
        dv = T.grad(cost, v)

        # Assign functions
        self.predict = theano.function([x], [o])
        self.predict_last = theano.function([x], [o_last])
        self.predict_class = theano.function([x, y], [prediction, e], allow_input_downcast=True)
        self.error = theano.function([x, y], e)
        self.ce_error = theano.function([x, y], cost, allow_input_downcast=True)
        self.bptt = theano.function([x, y], [dE, dU, dW, db, dV, dc], allow_input_downcast=True)

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mE = (decay * self.mE + (1 - decay) * dE ** 2).astype(theano.config.floatX)
        mU = (decay * self.mU + (1 - decay) * dU ** 2).astype(theano.config.floatX)
        mW = (decay * self.mW + (1 - decay) * dW ** 2).astype(theano.config.floatX)
        mV = (decay * self.mV + (1 - decay) * dV ** 2).astype(theano.config.floatX)
        mb = (decay * self.mb + (1 - decay) * db ** 2).astype(theano.config.floatX)
        mc = (decay * self.mc + (1 - decay) * dc ** 2).astype(theano.config.floatX)
        mv = (decay * self.mv + (1 - decay) * dv ** 2).astype(theano.config.floatX)

        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.In(decay, value=0.9)],
            [],
            updates=[(E, E - (learning_rate * dE / T.sqrt(mE + 1e-6)).astype(theano.config.floatX)),
                     (U, U - (learning_rate * dU / T.sqrt(mU + 1e-6)).astype(theano.config.floatX)),
                     (W, W - (learning_rate * dW / T.sqrt(mW + 1e-6)).astype(theano.config.floatX)),
                     (V, V - (learning_rate * dV / T.sqrt(mV + 1e-6)).astype(theano.config.floatX)),
                     (b, b - (learning_rate * db / T.sqrt(mb + 1e-6)).astype(theano.config.floatX)),
                     (c, c - (learning_rate * dc / T.sqrt(mc + 1e-6)).astype(theano.config.floatX)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc),
                     (self.mv, mv)
                     ], allow_input_downcast=True)

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(X[i:i + self.mini_batch_dim, :], Y[i:i + self.mini_batch_dim, :])
                       for i in range(0, np.shape(X)[0])])

    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        # num_words = np.shape(X)[0] * np.shape(X)[1]
        return self.calculate_total_loss(X, Y)

    def save(self, dir_name=None, filetag=None):
        self.train_time = int((time.time() - self.start_time) * 1000)
        if dir_name is None:
            dir_name = GRU_DATA_DIRECTORY + 'other' + '/'
        else:
            dir_name = GRU_DATA_DIRECTORY + dir_name + '/'

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

        if filetag is not None:
            filename = dir_name + filetag + '.npz'
        else:
            filename = dir_name + self.get_file_tag(-1, self.train_time) + '.npz'

        print("Saving model to file: " + filename)
        np.savez(filename,
                 E=self.E.get_value(),
                 U=self.U.get_value(),
                 W=self.W.get_value(),
                 V=self.V.get_value(),
                 b=self.b.get_value(),
                 c=self.c.get_value(),
                 train_time=self.train_time
                 )

    def load(self, filetag, dir_name=None):
        print("Starting model loading...")
        if dir_name is None:
            dir_name = GRU_DATA_DIRECTORY
        else:
            dir_name = GRU_DATA_DIRECTORY + dir_name + '/'

        if filetag is not None:
            filename = dir_name + filetag + ".npz"
        else:
            try:
                filename = dir_name + self.get_file_tag(-5, -5) + ".npz"
            except:
                raise Exception("Must insert a filetag with model.get_file_tag(#dataset,#epoch)")

        npzfile = np.load(filename)
        E, U, W, V, b, c, train_time = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], \
                                       npzfile["c"], npzfile["train_time"]
        # self.train_time.set_value(train_time)

        print("Building model from %s " % (
            filename))
        sys.stdout.flush()

        self.E.set_value(E[0, :])
        self.U.set_value(U)
        self.W.set_value(W)
        self.V.set_value(V[0])
        self.b.set_value(b)
        self.c.set_value(c)

    def get_file_tag(self, dataset, epoch):
        return 'GRU_{0}[{1}.{2}.{3}.{4}.{5}]'. \
            format(self.signal_name, self.signal_dim, self.hidden_dim, self.bptt_truncate, dataset, epoch)

    def train(self, X, Y, save_directory, batch_size=32, number_of_batches=None, window_size=256,
              learning_rate_val=0.05, nepoch_val=5000, decay=0.9, track_loss=False, first_index=0, save_distance=100):

        x_windows, y_end_values, N_batches, last_index = segment_signal(X, window_size, overlap=0.25, start_index=0)
        y_windows, y_end_values, N_batches, last_index = segment_signal(Y, window_size, overlap=0.25, start_index=0)

        window_indexes = np.random.permutation(N_batches)  # randomly select windows

        # first mini_batch_dim are training - 70%
        x_train = x_windows[window_indexes[0:batch_size], :]
        y_train = y_windows[window_indexes[0:batch_size], :]

        # last windows are testing - 30% -> implement on cross_validation
        # testing_X_batch = X[window_indexes[mini_batch_dim:int(mini_batch_dim + mini_batch_dim * (0.3 / 0.7))], :]
        # testing_Y_batch = Y[window_indexes[mini_batch_dim:int(mini_batch_dim + mini_batch_dim * (0.3 / 0.7))], :]


        if track_loss:
            plt.ion()

        print("Start Processing")

        t1 = time.time()
        self.train_with_sgd(x_train, y_train, nepoch_val, decay, track_loss, save_directory, 0,
                            save_distance=save_distance)

        print("Dataset trained in: ~%d seconds" % int(time.time() - t1))
        self.save(save_directory, self.get_file_tag(-5, -5))

    def train_signal(self, X, Y, signal2model, decay=0.95, track_loss=False, save_distance=1000):
        self.train(X, Y, signal2model.signal_directory,
                   signal2model.batch_size,
                   signal2model.number_of_batches,
                   signal2model.window_size,
                   signal2model.learning_rate_val,
                   signal2model.number_of_epochs,
                   decay=decay,
                   track_loss=track_loss,
                   save_distance=save_distance)

    def train_with_sgd(self, x_train, y_train, patience, decay, track_loss, save_directory, dataset=0,
                       save_distance=1000):
        loss = [self.calculate_loss(x_train, y_train)]
        lower_error_threshold, higher_error_threshold = [0.000001, 1]
        lower_learning_rate = 0.000000001
        count_to_break = 0
        count_up_slope = 0
        test_gradient = False
        if self.current_learning_rate <= lower_learning_rate:
            self.current_learning_rate = 0.00001

        for epoch in range(patience):
            t_epoch_1 = time.time()
            loss.append(self.calculate_loss(x_train, y_train))
            t_loss = time.time() - t_epoch_1

            if epoch > 2:
                if loss[-1] < np.min(loss[:-1]):
                    print("Loss: " + str(loss[-1]))
                relative_loss_gradient = (loss[-2] - loss[-1]) / (loss[-2] + loss[-1])
                if math.isnan(loss[-1]):
                    break
                if relative_loss_gradient < 0:
                    count_up_slope += 1
                    if count_up_slope >= 5:
                        count_up_slope = 0
                        if np.size(loss) > 10:
                            print("Min Loss in the last {0} epochs: {1:.3f} < {2:.3f} (x1000)?".format
                                  (10, np.min(loss[-5:]) * 1000, np.min(loss[-10:-5])) * 1000)
                            if np.min(loss[-5:]) > np.min(loss[-10:-5]):
                                self.current_learning_rate = self.current_learning_rate * 3 / 4
                                print("Adjusting learning rate: " + str(self.current_learning_rate))

                    count_to_break = 0
                elif relative_loss_gradient > higher_error_threshold:
                    self.current_learning_rate = self.current_learning_rate * 5 / 4
                    count_to_break = 0
                elif relative_loss_gradient < lower_error_threshold:
                    self.current_learning_rate = self.current_learning_rate * 3 / 4
                    test_gradient = True
                    count_to_break += 1
                    print("Adjusting learning rate to lower value: " + str(self.current_learning_rate))

                if loss[-1] == 0:
                    break

                if count_to_break > 5:
                    break

                elif test_gradient:
                    test_gradient = False

            if epoch % 10 == 0 and track_loss:
                plt.clf()
                plt.plot(loss[1:])
                plt.pause(0.05)
                if len(loss) > 100:
                    plt.plot(loss[-100:])
                    plt.plot(np.diff(loss[-101:]))

            t1 = time.time()
            if epoch % 10 == 0 or epoch == 0:
                print("In epoch %d of %d" % (epoch, patience))
            # For each training example...
            indexes = np.random.permutation(np.shape(y_train)[0])
            for i in range(0, len(indexes), self.mini_batch_dim):
                # One Minibatch SGD step
                ind = indexes[i:i + self.mini_batch_dim]
                self.sgd_step(x_train[ind, :], y_train[ind, :], self.current_learning_rate, decay)

            if epoch % 10 == 0 or epoch == 0:
                t2 = time.time()
                print("Time to calculate loss: {0} ".format(t_loss))
                print("SGD Step time: ~%d seconds" % int(t2 - t1))
                sys.stdout.flush()

            if epoch > 1 and epoch % save_distance == 0:
                self.save(dir_name=save_directory, filetag=self.get_file_tag(dataset, epoch))

            if epoch % 10 == 0 or epoch == 0:
                t2 = time.time()
                print("Epoch time: ~%d seconds" % int(t2 - t_epoch_1))
                sys.stdout.flush()

    def generate_predicted_signal(self, N, starting_signal, window_seen_by_GRU_size):
        print('Starting model generation')
        percent = 0
        new_signal = np.asarray(starting_signal)
        for i in range(N):
            if int(i * 100 / N) % 5 == 0:
                print('.', end='')
                sys.stdout.flush()
            elif int(i * 100 / N) % 20 == 0:
                percent += 0.2
                print('{0}%'.format(percent))
                sys.stdout.flush()

            signal = self.generate_online_predicted_signal(new_signal, window_seen_by_GRU_size)  # print(new_signal)
            # print(new_signal)
            new_signal = np.append(new_signal, signal)
            # sampled_word = word_to_index[unknown_token]

        # return self.generate_online_predicted_signal(starting_signal, window_seen_by_GRU_size)
        return new_signal

    def generate_online_predicted_signal(self, starting_signal, window_seen_by_GRU_size):
        new_signal = starting_signal

        if len(new_signal) <= window_seen_by_GRU_size:
            signal = new_signal
        else:
            signal = new_signal[-window_seen_by_GRU_size:]
            # print(signal)

        return self.predict_last(signal)
