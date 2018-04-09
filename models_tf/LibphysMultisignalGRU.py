from DeepLibphys.models_tf.LibphysDNN import *


class LibphysMultisignalGRU(LibphysDNN):
    def __init__(self, signal2model=None):
        super().__init__(signal2model)

    def get_common_variables(self):
        Hd = self.signal2model.hidden_dim
        Sd = self.signal2model.signal_dim + 1
        Ng = self.signal2model.n_grus
        Ns = self.signal2model.n_signals

        E = np.random.uniform(-np.sqrt(1. / Sd), np.sqrt(1. / Sd),
                              (Ns, Hd, Sd))

        U = np.random.uniform(-np.sqrt(1. / Hd), np.sqrt(1. / Hd),
                              (Ns, Ng, 3, Hd, Hd))
        W = np.random.uniform(-np.sqrt(1. / self.signal2model.hidden_dim), np.sqrt(1. / Hd),
                              (Ns, Ng, 3, Hd, Hd))
        V = np.random.uniform(-np.sqrt(1. / Hd), np.sqrt(1. / Hd),
                              (Ns, Sd, Hd))

        b = np.zeros((Ns, Ng, 3, Hd))
        c = np.zeros((Ns, Sd))

        self.X = tf.placeholder(tf.int32, shape=[Ns, None, None], name="X")
        self.Y = tf.placeholder(tf.int32, shape=[Ns, None, None], name="Y")
        self.identity = tf.eye(Sd)

        return E, U, W, V, b, c

    def get_specific_variables(self):
        Hd = self.signal2model.hidden_dim
        Ns = self.signal2model.n_signals

        CW = np.random.uniform(-np.sqrt(1. / (Hd * Hd)), np.sqrt(1. / (Hd * Hd)),
                               (3, (Hd * Ns), (Hd * Ns)))
        CU = np.random.uniform(-np.sqrt(1. / (Hd * Ns)), np.sqrt(1. / (Hd * Ns)),
                               (3, (Hd * Ns), (Hd * Ns)))
        Cb = np.zeros((3, Hd * Ns))

        self.CU = tf.Variable(CU, trainable=True, dtype=tf.float32, name="CU")
        self.CW = tf.Variable(CW, trainable=True, dtype=tf.float32, name="CW")
        self.Cb = tf.Variable(Cb, trainable=True, dtype=tf.float32, name="Cb")

        self.central_parameters = [self.CU, self.CW, self.Cb]
        self.trainables = self.parameters + self.central_parameters

    def GRUnn(self, out_prev, x_t):
        E, U, W, V, b, c, n_grus = self.E, self.U, self.W, self.V, self.b, self.c, self.signal2model.n_grus
        CU, CW, Cb = self.CU, self.CW, self.Cb,
        Hd, Sd, Bd = self.signal2model.hidden_dim, self.signal2model.signal_dim + 1, tf.shape(x_t[0])[0]
        coversion_ones = tf.ones((1, Bd), dtype=tf.float32, name="conversion_matrix")
        [S_prev, s_prev], _ = out_prev

        def GRU(last_input, gru_params):
            s_g_prev, u, w, b = gru_params

            z = tf.nn.sigmoid(tf.matmul(u[0], last_input) +
                              tf.matmul(w[0], s_g_prev) +
                              tf.matmul(tf.reshape(b[0], (tf.shape(w)[1], 1)), coversion_ones))

            r = tf.nn.sigmoid(tf.matmul(u[1], last_input) + tf.matmul(w[1], s_g_prev) +
                              tf.matmul(tf.reshape(b[1], (tf.shape(w)[2], 1)), coversion_ones))

            value = tf.matmul(u[2], last_input) + tf.matmul(w[2], s_g_prev * r) + \
                    tf.matmul(tf.reshape(b[2], (tf.shape(w)[2], 1)), coversion_ones)
            s_candidate = tf.nn.tanh(value)
            output = tf.add(((tf.ones_like(z) - z) * s_candidate), (z * s_g_prev), name="out_GRU")

            return output

        # x_e -> (Hd x Mb)
        x_e = tf.stack((tf.gather(E[0], x_t[0], axis=1), tf.gather(E[1], x_t[1], axis=1)), axis=0, name="x_e")

        # INPUT GRU LAYER
        s_0 = tf.stack((GRU(x_e[0], [s_prev[0, 0], U[0, 0], W[0, 0], b[0, 0]]),
                        GRU(x_e[0], [s_prev[0, 1], U[1, 0], W[1, 0], b[1, 0]])), axis=0, name="s_0")

        # CENTRAL PROCESSING UNIT GRU LAYER
        S = GRU(tf.concat((s_0[0], s_0[1]), axis=0), [S_prev, CU, CW, Cb])

        # OUTPUT GRU LAYER
        s_1 = tf.stack((GRU(S[:Hd], [s_prev[1, 0], U[0, 1], W[0, 1], b[0, 1]]),
                        GRU(S[Hd:], [s_prev[1, 1], U[1, 1], W[1, 1], b[1, 1]])), axis=0, name="s_1")

        logits_0 = tf.matmul(V[0], s_1[0]) + tf.matmul(tf.reshape(c[0], (Sd, 1)), coversion_ones)
        logits_1 = tf.matmul(V[1], s_1[1]) + tf.matmul(tf.reshape(c[1], (Sd, 1)), coversion_ones)

        o_t_0 = tf.nn.softmax(logits_0, axis=2)
        o_t_1 = tf.nn.softmax(logits_1, axis=2)

        s_t = [S, tf.stack((s_0, s_1))]
        o_t = tf.stack((o_t_0, o_t_1))
        return [s_t, o_t]  # , logits]

    def feed_forward_predict(self, X_batch):
        # initial_s has (n_grus x n_signals x Hd x Bd)
        initial_s = [tf.zeros((self.signal2model.hidden_dim * 2, tf.shape(X_batch[0])[1]), dtype=np.float32),
                     tf.zeros((self.signal2model.n_grus, self.signal2model.n_signals, self.signal2model.hidden_dim,
                               tf.shape(X_batch[0])[1]), dtype=np.float32)]
        # initial_s has (n_signals x Hd x Bd)
        initial_out = tf.zeros((self.signal2model.n_signals, self.signal2model.signal_dim + 1, tf.shape(X_batch[0])[1]),
                                dtype=np.float32)
        # initial_l = tf.zeros((self.signal2model.signal_dim, tf.shape(x_batch)[1]), dtype=np.float32)

        # x_batch = (N x Bd) - N (samples); Bd - Batch dimension
        # [s, o, l] = tf.scan(self.GRUnn, x_batch, initializer=[initial_s, initial_out, initial_l], parallel_iterations=1,
        #                     name="network_output")
        [_, o] = tf.scan(self.GRUnn, X_batch, initializer=[initial_s, initial_out],
                         parallel_iterations=1,
                         name="network_output")
        return o

    def calculate_mse(self):
        _y = self.to_one_hot_vector_in_mini_batches(self.Y)
        return tf.reduce_mean(tf.subtract(tf.transpose(self.out, perm=[2, 0, 1]), _y) ** 2, axis=2, name="mse")

    def feed_forward_predict_with_states(self, X_batch):
        initial_s = [tf.zeros((self.signal2model.hidden_dim * 2, tf.shape(X_batch[0])[1]), dtype=np.float32),
                     tf.zeros((self.signal2model.n_signals, self.signal2model.hidden_dim,
                               tf.shape(X_batch[0])[1]), dtype=np.float32)]
        initial_out = tf.zeros((self.signal2model.n_signals, self.signal2model.signal_dim, tf.shape(X_batch[0])[1]),
                                dtype=np.float32)
        # initial_l = tf.zeros((self.signal2model.signal_dim, tf.shape(x_batch)[1]), dtype=np.float32)

        # x_batch = (N x Bd) - N (samples); Bd - Batch dimension
        # [s, o, l] = tf.scan(self.GRUnn, x_batch, initializer=[initial_s, initial_out, initial_l], parallel_iterations=1,
        #                     name="network_output")
        [s, o] = tf.scan(self.GRUnn, X_batch, initializer=[initial_s, initial_out],
                         parallel_iterations=1,
                         name="network_output")
        return [s, o]

    def calculate_predictions(self):
        # MAP NOT WORKING:
        # shape(X)[0] -> Windows
        # shape(X)[1] -> Samples
        # n_batches = int(signal2model.batch_size / self.signal2model.mini_batch_size)
        # N = tf.shape(self.X)[1]
        # print(X)
        # get the matrices from E with tf.gather(E, X, axis=1, name="X_e")
        # transpose these matrices for (batch_size, HD, N)
        # reshape to enter map, where each minibatch is entered at the same time (n_batches, mini_batch, HD, N)
        # transpose to enter the DNN inside -> (n_batches, N, mini_batch)

        return self.feed_forward_predict(tf.transpose(self.X, perm=[2, 0, 1]))

    def calculate_mse(self):
        _y = self.to_one_hot_vector_in_mini_batches(self.Y)
        return tf.reduce_sum(tf.reduce_mean((tf.subtract(tf.transpose(self.out, perm=[1, 3, 0, 2]), _y) ** 2), axis=3,
                              name="mse"), name="mse_sum", axis=0)

    def calculate_mse_loss(self):
        return tf.reduce_mean(self.calculate_minibatch_mse(), axis=0, name="loss")

    def to_one_hot_vector_in_mini_batches(self, tensor):
        return self.get_one_hot(tensor)

    def get_one_hot(self, columns):
        return tf.gather(self.identity, columns)

    def shuffle(self, x, y, random_indexes):
        return {self.X: x[:, random_indexes],
                self.Y: y[:, random_indexes]
                }

    def train(self, X0, Y0, X1, Y1, signal2model=None):
        self.batch_size += np.shape(X1)[0]
        self.init_time = time.time()
        plt.ion()
        if signal2model is not None:
            self.signal2model = signal2model
        plt.ion()
        condition_not_met = True
        history = []
        self.epoch = 0
        self.loss_history = []
        tf.summary.scalar('loss', self.loss)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('train',
                                             self.session.graph)
        # run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        tf.global_variables_initializer()
        while condition_not_met:
            # epoch += 1
            self.epoch += 1
            tic = time.time()
            n_batches = int(self.signal2model.batch_size / self.signal2model.mini_batch_size)
            random_indexes = np.random.permutation(self.signal2model.batch_size)
            groups = np.reshape(random_indexes,
                                (n_batches,
                                 self.signal2model.mini_batch_size))

            nones = np.random.permutation(random_indexes)[:self.signal2model.mini_batch_size]
            x0 = np.copy(X0)
            x1 = np.copy(X1)
            x0[nones[:int(self.signal2model.mini_batch_size / 2)]] = \
                np.ones_like(x0[nones[:int(self.signal2model.mini_batch_size / 2)]]) * self.signal2model.signal_dim
            x1[nones[int(self.signal2model.mini_batch_size / 2):]] = \
                np.ones_like(x0[nones[:int(self.signal2model.mini_batch_size / 2)]]) * self.signal2model.signal_dim

            X = np.stack((x0, x1))
            Y = np.stack((Y0, Y1))


            for group in groups:
                dictionary = self.shuffle(X, Y, group)
                group_loss, op = self.session.run(
                    [self.loss_op, self.optimize_op],
                    feed_dict=dictionary)  # , options=run_options)

                # print("toc: {0} secs".format(time.time()-tic))

            new_tic = time.time()
            info, full_loss = self.session.run(
                [merged, self.loss],
                {self.X: X,
                 self.Y: Y}
            )
            plt.clf()
            self.loss_history.append(full_loss)
            if len(history) > 20:
                plt.plot(ni.smooth(np.array(self.loss_history), 20, window="flat"))
            plt.plot(self.loss_history)
            plt.pause(0.02)
            # print("loss toc: {0} secs".format(time.time() - new_tic))
            train_writer.add_summary(info, self.epoch)
            print(full_loss)

            condition_not_met = self.calculate_learning_rate_and_control_sequence()
            # # print(condition_not_met)
            # history.append(full_loss)
            # plt.clf()
            # plt.plot(history)
            # if len(history) > 20:
            #     plt.plot(ni.smooth(np.array(history), 20, window="flat"))
            # plt.pause(0.01)
            # print(self.loss)

        self.train_time = self.init_time - time.time()

    def generate_online_predicted_signal(self, X0, X1):
        x = tf.stack((tf.Variable(X0, dtype=tf.int32), tf.Variable(X1, dtype=tf.int32)))
        self.session.run(tf.global_variables_initializer())
        return self.session.run(self.feed_forward_predict(x))

    def generate_predicted_signal(self, N=None, X0=None, X1=None, window_seen_by_GRU_size=256, uncertainty=0.001):
        # We start the sentence with the start token
        GENERATE_X0 = (X0 is None)
        GENERATE_X1 = (X1 is None)

        if GENERATE_X0:
            new_X0 = np.reshape(np.array([np.random.randint(self.signal2model.signal_dim) for i in range(3)]),
                                (3, 1))
        else:
            new_X0 = np.array([X0[0]])
            N = np.shape(X0)[-1]
            print("Dimension must be equal")

        if GENERATE_X1:
            new_X1 = np.reshape(np.array([0 for i in range(3)]), (3, 1))
        else:
            new_X1 = np.array([X1[0]])
            if N is None or N > np.shape(X1):
                N = np.shape(X1)[-1]

        # Repeat until we get an end token
        print('Starting model generation')
        percent = 0
        plt.ion()
        for i in range(N):
            if len(new_X0) > window_seen_by_GRU_size:
                signal0 = new_X0[-window_seen_by_GRU_size:]
                signal1 = new_X1[-window_seen_by_GRU_size:]
            else:
                signal0 = new_X0
                signal1 = new_X1

            if int(i * 100 / N) % 5 == 0:
                print('.', end='')
            elif int(i * 100 / N) % 20 == 0:
                percent += 0.2
                print('{0}%'.format(percent))
            [p1, p2] = self.generate_online_predicted_signal(signal0, signal1)
            next_sample_probs_1 = np.asarray(p1[:, :, -1], dtype=float)
            next_sample_probs_1 = next_sample_probs_1 / \
                                  (np.reshape(np.sum(next_sample_probs_1, axis=1), (np.shape(signal0)[0], 1))
                                   * np.ones((1, self.signal2model.signal_dim + 1)))
            next_sample_probs_2 = np.asarray(p2[:, :, -1], dtype=float)
            next_sample_probs_2 = next_sample_probs_2 / \
                                  (np.reshape(np.sum(next_sample_probs_2, axis=1), (np.shape(signal1)[0], 1))
                                   * np.ones((1, self.signal2model.signal_dim + 1)))

            n_0 = np.array([np.random.choice(np.arange(self.signal2model.signal_dim + 1), p=next_sample_prob)
                            for next_sample_prob in next_sample_probs_1])
            n_1 = np.array([np.random.choice(np.arange(self.signal2model.signal_dim + 1), p=next_sample_prob)
                            for next_sample_prob in next_sample_probs_2])

            new_X0 = np.array(new_X0.T.tolist() + [n_0]).T
            new_X1 = np.array(new_X1.T.tolist() + [n_1]).T
            if not GENERATE_X0:
                new_X0[-1] = X0[i]
            if not GENERATE_X1:
                new_X1[-1] = X1[i]

            plt.clf()
            plt.plot(new_X0[0])
            plt.plot(new_X1[0])
            plt.pause(0.01)

        return new_X0, new_X1