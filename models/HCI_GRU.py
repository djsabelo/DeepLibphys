import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import theano as theano
import theano.tensor as T
from theano import printing

from DeepLearningTryAndError.utils import make_training_sets

GRU_DATA_DIRECTORY = "../data/trained/"


class LibPhys_GRU:
    def __init__(self, signal_dim, hidden_dim=128, signal_name=None):

        # For staters, x and y have the same dimension (100% x 100%)
        self.x_dim = signal_dim[0]
        self.y_dim = signal_dim[0]
        self.hidden_dim = hidden_dim
        self.bptt_truncate = -1
        self.signal_name = signal_name
        self.current_learning_rate = 0.01
        self.start_time = 0
        self.train_time = 0
        # Initialize the network parameters
        E_x = np.random.uniform(-np.sqrt(1. / self.x_dim), np.sqrt(1. / self.x_dim),
                                (2, hidden_dim, self.x_dim))
        E_y = np.random.uniform(-np.sqrt(1. / self.y_dim), np.sqrt(1. / self.signal_dim),
                                (2, hidden_dim, self.signal_dim))
        U = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (6, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (6, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (2, self.signal_dim, hidden_dim))
        b = np.zeros((6, hidden_dim))
        c = np.zeros(2, self.signal_dim)
        # Theano: Created shared variables
        self.E_x = theano.shared(name='Ex', value=E_x.astype(theano.config.floatX))
        self.E_y = theano.shared(name='Ey', value=E_x.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        # SGD / rmsprop: Initialize parameters
        self.mE_x = theano.shared(name='mEx', value=np.zeros(E_x.shape).astype(theano.config.floatX))
        self.mE_y = theano.shared(name='mEy', value=np.zeros(E_y.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
        self.conversion_ones = theano.shared(name='ones', value=np.ones((self.mini_batch_size, 1)))

        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        E_x, E_y, V, U, W, b, c = self.E_x, self.E_y, self.V, self.U, self.W, self.b, self.c

        x = T.ivector('x')
        y = T.ivector('y')
        x_label = T.ivector('x')
        y_label = T.ivector('y')

        def forward_prop_step(x_t, y_t, s_t_prev):
            # Word embedding layer
            x_e = E_x[:, x_t]
            y_e = E_y[:, y_t]

            def GRU(i, U, W, b, x_0, s_prev):
                pb = printing.Print('b')

                b1 = T.specify_shape((self.conversion_ones * b[i * 3, :]).T, T.shape(x_0))
                b2 = T.specify_shape((self.conversion_ones * b[i * 3 + 1, :]).T, T.shape(x_0))
                b3 = T.specify_shape((self.conversion_ones * b[i * 3 + 2, :]).T, T.shape(x_0))

                z = T.nnet.hard_sigmoid(U[i * 3 + 0].dot(x_0) + W[i * 3 + 0].dot(s_prev) + b1)
                r = T.nnet.hard_sigmoid(U[i * 3 + 1].dot(x_0) + W[i * 3 + 1].dot(s_prev) + b2)
                c = T.tanh(U[i * 3 + 2].dot(x_0) + W[i * 3 + 2].dot(s_prev * r) + b3)

                return (T.ones_like(z) - z) * c + z * s_prev

            p_o = printing.Print('juju')

            # GRU Layer 1
            s[0] = GRU(0, U, W, b, x_e, s_t_prev[0])

            # GRU Layer 2
            s[1] = GRU(1, U, W, b, y_e, s_t_prev[1])

            c_matrix = (self.conversion_ones * c).T
            juju = V.dot(s) + c_matrix
            ot = printing.Print("o_t")
            o_t = T.nnet.softmax(juju.T).T

            return [o_t, s]

        [o, s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=T.zeros(self.hidden_dim)),
                          dict(initial=T.zeros(self.hidden_dim)),
                          dict(initial=T.zeros(self.hidden_dim))])

        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        p_o = printing.Print('o_error')
        # Total cost (could add regularization here)
        cost = p_o(o_error)

        # Gradients
        dE_x = T.grad(cost, E_x)
        dE_y = T.grad(cost, E_y)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)

        # Assign functions
        self.predict = theano.function([x], [o], allow_input_downcast=True)
        self.predict_class = theano.function([x], prediction, allow_input_downcast=True)
        self.ce_error = theano.function([x, y], cost, allow_input_downcast=True)
        self.bptt = theano.function([x, y], [dE_x, dE_y, dU, dW, db, dV, dc], allow_input_downcast=True)

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mE_x = decay * self.mE_x + (1 - decay) * dE_x ** 2
        mE_y = decay * self.mE_y + (1 - decay) * dE_y ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.In(decay, value=0.9)],
            [],
            updates=[(E_x, E_x - learning_rate * dE_x / T.sqrt(dE_x + 1e-6)),
                     (E_y, E_y - learning_rate * dE_y / T.sqrt(dE_y + 1e-6)),
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mE_x, mE_x),
                     (self.mE_y, mE_y),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                     ], allow_input_downcast=True)

    def calculate_total_loss(self, X, Y, X_labels, Y_labels):
        return np.sum(
            [self.ce_error([x, y], [x_label, y_label]) for x, y, x_label, y_label in zip(X, Y, X_labels, Y_labels)])

    def calculate_loss(self, X, Y, X_labels, Y_labels):
        return self.calculate_total_loss(X, Y) / float(self.x_dim)

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
        elif self.signal_name is None:
            filename = dir_name + "biosig_libPHYS_GRU_dict_dim-" + str(
                self.signal_dim) + "_hidden_dim-" + str(
                self.hidden_dim) + "_bptt_truncate-" + str(self.bptt_truncate) + ".npz"
        else:
            filename = dir_name + "biosig_libPHYS_GRU_" + self.signal_name + "dict_dim-" + str(
                self.signal_dim) + "_hidden_dim-" + str(
                self.hidden_dim) + "_bptt_truncate-" + str(self.bptt_truncate) + ".npz"
        print("Saving model to file: " + filename)
        np.savez(filename,
                 E=self.E.get_value(),
                 U=self.U.get_value(),
                 W=self.W.get_value(),
                 V=self.V.get_value(),
                 b=self.b.get_value(),
                 c=self.c.get_value(),
                 train_time=self.train_time,
                 num_examples_seen=self.num_examples_seen
                 )

    def load(self, signal_name=None, filetag=None, dir_name=None):
        print("Starting sinal loading...")
        if dir_name is None:
            dir_name = GRU_DATA_DIRECTORY
        else:
            dir_name = GRU_DATA_DIRECTORY + dir_name + '/'

        if filetag is not None:
            filename = dir_name + filetag + ".npz"
        elif signal_name is None:
            filename = dir_name + "biosig_libPHYS_GRU_dict_dim-" + str(
                self.signal_dim) + "_hidden_dim-" + str(
                self.hidden_dim) + "_bptt_truncate-" + str(self.bptt_truncate) + ".npz"
        else:
            # self.signal_name = signal_name
            filename = dir_name + "biosig_libPHYS_GRU_" + signal_name + "dict_dim-" + str(
                self.signal_dim) + "_hidden_dim-" + str(
                self.hidden_dim) + "_bptt_truncate-" + str(self.bptt_truncate) + ".npz"
        npzfile = np.load(filename)
        E, U, W, V, b, c = [], [], [], [], [], []
        try:
            E, U, W, V, b, c, num_examples_seen = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], \
                                                  npzfile["c"], npzfile["num_examples_seen"]
            self.num_examples_seen.set_value(num_examples_seen)
            hidden_dim, signal_dim = E.shape[0], E.shape[1]
            print("Building model from %s with hidden_dim=%d signal_dim=%d and number_examples" % (
                filename, hidden_dim, signal_dim, num_examples_seen))
        except:
            E, U, W, V, b, c = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], \
                               npzfile["c"]
            hidden_dim, signal_dim = E.shape[0], E.shape[1]
            print("Building model from %s with hidden_dim=%d signal_dim=%d" % (
                filename, hidden_dim, signal_dim))

        try:
            self.train_time = npzfile["train_time"]
        except:
            pass

        sys.stdout.flush()
        self.E.set_value(E)
        self.U.set_value(U)
        self.W.set_value(W)
        self.V.set_value(V)
        self.b.set_value(b)
        self.c.set_value(c)

    def get_file_tag(self, dataset, epoch):
        return 'GRU_{0}[{1}.{2}.{3}.{4}.{5}]'. \
            format(self.signal_name, self.signal_dim, self.hidden_dim, self.bptt_truncate, dataset, epoch)

    def train_for_crossvalidation(self, X, Y, save_directory, filename, batch_size=32, number_of_batches=None,
                                  window_size=256, learning_rate_val=0.001, nepoch_val=50, decay=0.9, track_loss=False):
        """Training with train and save test data for later
        """
        hidden_dim = self.E.shape[0]
        signal_dim = self.E.shape[1]
        self.start_time = time.time()
        if number_of_batches is None or batch_size * number_of_batches * window_size > len(X):
            number_of_batches = int(len(X) / (window_size * batch_size))

        n_windows = batch_size * number_of_batches

        X = X[0:n_windows * window_size]  # the last window frame is rejected
        Y = Y[0:n_windows * window_size]

        self.current_learning_rate = learning_rate_val
        X = np.reshape(X, (n_windows, window_size))
        Y = np.reshape(Y, (n_windows, window_size))

        train_indexes, test_indexes, train, test = make_training_sets(X, 0.7)

        x_train = train
        y_train = Y[train_indexes]

        x_test = test
        y_test = Y[test_indexes]

        # np.savez(GRU_DATA_DIRECTORY + save_directory + '/' + filename, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        print("Saved values to %s." % (GRU_DATA_DIRECTORY + save_directory + '/' + filename))

        if track_loss:
            plt.plot(x_train[0, :])
            plt.show()
            plt.ion()

        n_train_windows = np.shape(x_train)[0]
        i = 0
        for i in range(0, n_train_windows, batch_size):
            end_index = i + batch_size
            if end_index >= n_train_windows:
                end_index = n_train_windows

            t1 = time.time()
            self.train_time = int(t1 - self.start_time)
            print("dataset from #" + str(i) + " to #" + str(end_index - 1))

            self.current_learning_rate = learning_rate_val
            # try:
            if i % batch_size * 10 == 0:
                signal_name = self.signal_name
                self.save(save_directory, self.get_file_tag(i, 0))

            self.train_with_sgd(x_train[i:end_index, :], y_train[i:end_index, :],
                                nepoch_val, decay, track_loss, save_directory, dataset=i)

            print("Dataset trained in: ~%d seconds" % int(time.time() - t1))
            # except:
            #     print("Error has occured in dataset #" + str(i))
        self.save(filetag=self.get_file_tag(i, 2000), dir_name=save_directory)

    def train(self, X, Y, save_directory, batch_size=32, number_of_batches=None, window_size=256,
              learning_rate_val=0.05, nepoch_val=5000, decay=0.9, track_loss=False, first_index=0, save_distance=100):
        hidden_dim = self.E.shape[0]
        signal_dim = self.E.shape[1]

        if number_of_batches is None or batch_size * number_of_batches * window_size > len(X):
            number_of_batches = int(len(X) / (window_size * batch_size))

        if number_of_batches == 0:
            number_of_batches = 1

        n_windows = batch_size * number_of_batches
        end_index = first_index
        self.current_learning_rate = learning_rate_val
        for i in range(0, n_windows, batch_size):
            first_index = end_index
            end_index = first_index + window_size * batch_size

            if end_index > n_windows * window_size:
                batch_size = n_windows - i
                end_index = batch_size * window_size

            matrix_N_shape = batch_size  # batch size, except for the last batch
            matrix_M_shape = window_size

            x_train = np.reshape(X[first_index: end_index], (matrix_N_shape, matrix_M_shape))
            y_train = np.reshape(Y[first_index: end_index], (matrix_N_shape, matrix_M_shape))

            if track_loss:
                plt.plot(x_train[0, :])
                plt.show()
                plt.ion()

            print("dataset from #" + str(i) + " to #" + str(i + batch_size - 1))
            # try:
            t1 = time.time()
            self.train_with_sgd(x_train, y_train, nepoch_val, decay, track_loss, save_directory, i,
                                save_distance=save_distance)
            print("Dataset trained in: ~%d seconds" % int(time.time() - t1))
            self.save(save_directory, self.get_file_tag(i, -5))

            # except:
            #     print("Error has occured in dataset #" + str(i))
        self.save(save_directory, self.get_file_tag(-5, -5))

    def train_signal(self, X, Y, signal, decay=0.95, track_loss=False, save_distance=100):
        self.train(X, Y, signal.signal_directory, signal.batch_size, signal.number_of_batches, signal.window_size,
                   signal.learning_rate_val, signal.number_of_epochs, decay=decay, track_loss=track_loss,
                   save_distance=save_distance)

    def train_signals(self, X, Y, signal2model, decay=0.9, track_loss=False, save_distance=100):
        hidden_dim = self.E.shape[0]
        signal_dim = self.E.shape[1]
        self.current_learning_rate = signal2model.learning_rate_val

        signal_batch = int(signal2model.batch_size / len(X))
        real_batch_size = signal_batch * len(X)
        y_train = np.zeros((real_batch_size, signal2model.window_size))
        x_train = np.zeros((real_batch_size, signal2model.window_size))
        random_indexes = np.random.permutation(real_batch_size)

        for i in range(len(X)):
            x_0, y_0 = self.make_batch(X[i], Y[i], signal_batch, window_size=signal2model.window_size)
            y_train[random_indexes[i * signal_batch:i * signal_batch + signal_batch], :] = y_0
            x_train[random_indexes[i * signal_batch:i * signal_batch + signal_batch], :] = x_0

        if track_loss:
            plt.show()
            plt.ion()

        print("dataset from #" + str(i) + " to #" + str(i + signal2model.batch_size - 1))
        # try:
        t1 = time.time()
        self.train_with_sgd(x_train, y_train, signal2model.number_of_epochs, decay, track_loss,
                            signal2model.signal_directory, 0, save_distance=save_distance)
        print("Dataset trained in: ~%d seconds" % int(time.time() - t1))
        self.save(signal2model.save_directory, self.get_file_tag(i, -5))

        # except:
        #     print("Error has occured in dataset")
        self.save(signal2model.signal_directory, self.get_file_tag(-5, -5))

    def make_batch(self, X, Y, batch_size, first_index=0, window_size=256):
        end_index = first_index + window_size * batch_size

        if end_index > batch_size * window_size:
            batch_size = int((end_index - first_index) / window_size)
            end_index = first_index + batch_size * window_size

        matrix_N_shape = batch_size  # batch size, except for the last batch
        matrix_M_shape = window_size

        x_train = np.reshape(X[first_index: end_index], (matrix_N_shape, matrix_M_shape))
        y_train = np.reshape(Y[first_index: end_index], (matrix_N_shape, matrix_M_shape))

        return x_train, y_train

    def train_with_sgd(self, x_train, y_train, nepoch_val, decay, track_loss, save_directory, dataset=0,
                       save_distance=10):
        num_examples_seen = 0
        loss = [self.calculate_loss(x_train, y_train)]
        lower_error_threshold, higher_error_threshold = [0.000001, 1]
        lower_learning_rate = 0.000000001
        count_to_break = 0
        count_up_slope = 0
        test_gradient = False
        if self.current_learning_rate <= lower_learning_rate:
            self.current_learning_rate = 0.00001

        for epoch in range(nepoch_val):
            t_epoch_1 = time.time()

            loss.append(self.calculate_loss(x_train, y_train))
            # if epoch == 1 and loss[0] > 1:
            #     self.current_learning_rate = 0.01
            #     print("Adjusting learing rate to higher value: " + str(self.current_learning_rate))
            if epoch > 2:
                print("Loss: " + str(loss[-1]))
                track_loss_in_this_epoch = False
                relative_loss_gradient = (loss[-2] - loss[-1]) / (loss[-2] + loss[-1])
                if math.isnan(loss[-1]):
                    break
                if relative_loss_gradient < 0:
                    count_up_slope += 1
                    if count_up_slope >= 5:
                        count_up_slope = 0
                        if np.size(loss) > 3:
                            print("Min Loss in the last {0} epochs: {1:.3f} < {2:.3f} ?".format
                                  (4, np.min(loss[-3:]), np.min(loss)))
                            if np.min(loss[-4:]) > np.min(loss):
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

                if count_to_break > 5:
                    break

                elif test_gradient:
                    test_gradient = False

            if epoch % 10 == 0 and track_loss:
                plt.clf()
                plt.plot(loss[1:])
                plt.pause(0.05)

            t1 = time.time()
            if epoch % 10 == 0 or epoch == 0:
                print("In epoch %d of %d" % (epoch, nepoch_val))
            # For each training example...
            for i in np.random.permutation(len(y_train)):
                # One SGD step
                self.sgd_step(x_train[i], y_train[i], self.current_learning_rate, decay)
                self.num_examples_seen += 1

            if epoch % 10 == 0 or epoch == 0:
                t2 = time.time()
                print("Seen " + str(self.num_examples_seen))
                print("SGD Step time: ~%d seconds" % int(t2 - t1))
                sys.stdout.flush()

            if epoch > 1 and epoch % save_distance == 0:
                self.save(dir_name=save_directory, filetag=self.get_file_tag(dataset, epoch))

            if epoch % 10 == 0 or epoch == 0:
                t2 = time.time()
                print("Epoch time: ~%d seconds" % int(t2 - t_epoch_1))
                sys.stdout.flush()

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
            if int(i * 100 / N) % 5 == 0:
                print('.', end='')
            elif int(i * 100 / N) % 20 == 0:
                percent += 0.2
                print('{0}%'.format(percent))
            new_signal.append(self.generate_online_predicted_signal(starting_signal, window_seen_by_GRU_size))
            # sampled_word = word_to_index[unknown_token]

        return new_signal[1:]

    def generate_online_predicted_signal(self, starting_signal, window_seen_by_GRU_size):
        # We start the sentence with the start token
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
