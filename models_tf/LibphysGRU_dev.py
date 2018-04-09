from DeepLibphys.models_tf.LibphysDNN import *



class LibphysGRU(LibphysDNN):
    def __init__(self, signal2model=None):
        super().__init__(signal2model)

    def get_specific_variables(self):
        self.trainables = self.parameters

    def get_common_variables(self):
        Hd = self.signal2model.hidden_dim
        Sd = self.signal2model.signal_dim
        Ng = self.signal2model.n_grus

        E = np.random.uniform(-np.sqrt(1. / Sd), np.sqrt(1. / Sd),
                              (Hd, Sd))

        U = np.random.uniform(-np.sqrt(1. / Hd), np.sqrt(1. / Hd),
                              (Ng, 3, Hd, Hd))
        W = np.random.uniform(-np.sqrt(1. / self.signal2model.hidden_dim), np.sqrt(1. / Hd),
                              (Ng, 3, Hd, Hd))
        V = np.random.uniform(-np.sqrt(1. / Hd), np.sqrt(1. / Hd),
                              (Sd, Hd))

        b = np.zeros((Ng, 3, Hd))
        c = np.zeros(Sd)
        self.identity = tf.eye(Sd)

        return E, U, W, V, b, c, [None, None]

    def GRUnn(self, out_prev, x_t):
        E, U, W, V, b, c = self.E, self.U, self.W, self.V, self.b, self.c
        Hd, Sd, Bd = self.signal2model.hidden_dim, self.signal2model.signal_dim, tf.shape(x_t)[0]
        coversion_ones = tf.ones((1, Bd), dtype=tf.float32, name="conversion_matrix")
        # s_prev, o_prev, l_prev = out_prev
        s_prev, o_prev = out_prev

        def GRU(last_input, gru_params):
            s_g_prev, u, w, b = gru_params

            z = tf.nn.sigmoid(tf.matmul(u[0], last_input) +
                              tf.matmul(w[0], s_g_prev) +
                              tf.matmul(tf.reshape(b[0], (Hd, 1)), coversion_ones))

            r = tf.nn.sigmoid(tf.matmul(u[1], last_input) + tf.matmul(w[1], s_g_prev) +
                              tf.matmul(tf.reshape(b[1], (Hd, 1)), coversion_ones))

            value = tf.matmul(u[2], last_input) + tf.matmul(w[2], s_g_prev * r) + \
                    tf.matmul(tf.reshape(b[2], (Hd, 1)), coversion_ones)
            s_candidate = tf.nn.tanh(value)
            output = tf.add(((tf.ones_like(z) - z) * s_candidate), (z * s_g_prev), name="out_GRU")

            return output

        # x_e -> (Hd x Mb)
        x_e = tf.gather(self.E, x_t, axis=1)
        s_t_ = []

        s_t_.append(GRU(x_e, [s_prev[0], U[0], W[0], b[0]]))
        s_t_.append(GRU(s_t_[0], [s_prev[1], U[1], W[1], b[1]]))
        s_t_.append(GRU(s_t_[1], [s_prev[2], U[2], W[2], b[2]]))
        s_t = tf.stack(s_t_)
        # tf.scan(GRU, (s_prev, self.U, self.W, self.b), initializer=x_e, parallel_iterations=1, name="states")

        logits = tf.matmul(self.V, s_t[-1]) + tf.matmul(tf.reshape(self.c, (Sd, 1)), coversion_ones)

        o_t = tf.nn.softmax(logits, axis=2)

        return [s_t, o_t]#, logits]

    def feed_forward_predict(self, x_batch):
        initial_s = tf.zeros((self.signal2model.n_grus, self.signal2model.hidden_dim, tf.shape(x_batch)[1]), dtype=np.float32)
        initial_out = tf.zeros((self.signal2model.signal_dim, tf.shape(x_batch)[1]), dtype=np.float32)
        # initial_l = tf.zeros((self.signal2model.signal_dim, tf.shape(x_batch)[1]), dtype=np.float32)

        # x_batch = (N x Bd) - N (samples); Bd - Batch dimension
        # [s, o, l] = tf.scan(self.GRUnn, x_batch, initializer=[initial_s, initial_out, initial_l], parallel_iterations=1,
        #                     name="network_output")
        [_, o] = tf.scan(self.GRUnn, x_batch, initializer=[initial_s, initial_out], parallel_iterations=1,
                            name="network_output")
        return o

    def feed_forward_predict_with_states(self, x_batch):
        initial_s = tf.zeros((self.signal2model.n_grus, self.signal2model.hidden_dim, tf.shape(x_batch)[1]),
                             dtype=np.float32)
        initial_out = tf.zeros((self.signal2model.signal_dim, tf.shape(x_batch)[1]), dtype=np.float32)
        # initial_l = tf.zeros((self.signal2model.signal_dim, tf.shape(x_batch)[1]), dtype=np.float32)

        # x_batch = (N x Bd) - N (samples); Bd - Batch dimension
        # [s, o, l] = tf.scan(self.GRUnn, x_batch, initializer=[initial_s, initial_out, initial_l], parallel_iterations=1,
        #                     name="network_output")
        [s, o] = tf.scan(self.GRUnn, x_batch, initializer=[initial_s, initial_out], parallel_iterations=1,
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

        return self.feed_forward_predict(tf.transpose(self.X))

    def to_one_hot_vector_in_mini_batches(self, matrix):
        return self.get_one_hot(matrix)

    def get_one_hot(self, columns):
        return tf.gather(self.identity, columns)

    def calculate_cross_entropy(self):
        return None
        # logits = tf.transpose(self.logits, perm=[2, 0, 1])
        # n_batches = int(self.signal2model.batch_size / self.signal2model.mini_batch_size)
        # y = tf.reshape(self.Y, (n_batches, self.signal2model.mini_batch_size, tf.shape(self.Y)[1]))
        # self.full_loss = tf.losses.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        # return self.full_loss

    def calculate_mse(self):
        _y = self.to_one_hot_vector_in_mini_batches(self.Y)
        return tf.reduce_mean(tf.subtract(tf.transpose(self.out, perm=[2, 0, 1]), _y) ** 2, axis=2, name="mse")

    def calculate_mse_vector_loss(self, x, y):
        with tf.variable_scope('vector_loss'):
            return tf.reduce_mean(self.calculate_minibatch_mse(x, y), axis=0, name="vector_loss")

    def calculate_mse_loss(self):
        return tf.reduce_mean(self.calculate_minibatch_mse(), axis=0, name="loss")

    @property
    def loss_op(self):
        """ An Operation that takes one optimization step. """
        return self.loss

    def init_optimizer(self):
        trainables = self.parameters
        grads = tf.gradients(self.loss, trainables)
        grad_var_pairs = zip(grads, trainables)
        # grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
        # self.learning_rate = tf.train.exponential_decay(
        #     self.signal2model.learning_rate_val, self.epoch, self.signal2model.count_to_break_max,
        #     self.signal2model.decay, staircase=True)
        # with tf.device('/gpu:1'):
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate_gpu)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self._optimize_op = optimizer.apply_gradients(grad_var_pairs)

    @property
    def optimize_op(self):
        """ An Operation that takes one optimization step. """
        return self._optimize_op

    def train(self, X, Y, signal2model=None):
        self.batch_size += np.shape(X)[0]
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

        # db_url = 'postgres://belo:passsword@localhost/postgres'
        # experiments, steps, model_params = labnotebook.initialize(db_url)
        # model_desc = {'loss': 0.}
        # experiment = labnotebook.start_experiment(model_desc=model_desc)
        tf.global_variables_initializer()
        while condition_not_met:
            self.epoch += 1
            # tic = time.time()
            random_indexes = np.random.permutation(self.signal2model.batch_size)
            groups = np.reshape(random_indexes,
                                (int(self.signal2model.batch_size/self.signal2model.mini_batch_size),
                                 self.signal2model.mini_batch_size))
            for group in groups:
                dictionary = self.shuffle(X, Y, group)
                op, group_loss = self.session.run(
                    [self.optimize_op, self.loss_op],
                    feed_dict=dictionary)#, options=run_options)
                # labnotebook.step_experiment(experiment,
                #                             timestep=str(self.epoch),
                #                             trainacc=0,
                #                             valacc=0,
                #                             trainloss=str(group_loss))
                # print("toc: {0} secs".format(time.time()-tic))


            # new_tic = time.time()
            full_loss = self.session.run(
                self.loss_op,
                {self.X: X,
                 self.Y: Y}
            )
            self.loss_history.append(full_loss)
            # labnotebook.step_experiment(experiment,
            #                             timestep=str(self.epoch),
            #                             trainacc=0,
            #                             valacc=0,
            #                             trainloss=str(group_loss),
            #                             custom_fields={'train time': self.train_time,
            #                                            "full loss": full_loss})
            plt.clf()
            if len(self.loss_history) > 20:
                plt.plot(ni.smooth(np.array(self.loss_history), 20, window="flat"))
            plt.plot(self.loss_history)
            plt.ylim([0, np.max(self.loss_history)])
            plt.pause(0.02)
            # print("loss toc: {0} secs".format(time.time() - new_tic))
            # train_writer.add_summary(info, epoch)
            # print(full_loss)

            condition_not_met = self.calculate_learning_rate_and_control_sequence()
            # print(condition_not_met)
            # condition_not_met = self.signal2model.number_of_epochs > epoch
            # # print(condition_not_met)
            # history.append(full_loss)
            # plt.clf()
            # plt.plot(history)
            # if len(history) > 20:
            #     plt.plot(ni.smooth(np.array(history), 20, window="flat"))
            # plt.pause(0.01)
            # print(self.loss)

        self.train_time = self.start_time - time.time()
        plt.figure()
        plt.plot(self.loss_history)
        plt.show()
        return True
        # labnotebook.end_experiment(experiment,
        #                            final_trainloss=full_loss)

    @staticmethod
    def load_full_model(self, model_name, dir_name, hidden_dim, signal_dim, dataset=-5, epoch=-5):
        """
        Loads the model

        :param dir_name: -string - directory name where the corresponding to the model for loading is
                            -> may use model.get_directory_tag(dataset, epoch)

        :param file_tag: - string - file_tag corresponding to the model for loading
                            -> use model.get_file_tag(dataset, epoch)
                            if given None it will assume that is the last version of the model get_file_tag(-5,-5)
        :return: None
        """

        print("Starting sinal loading...")

        file_tag = self.get_static_file_tag(model_name, signal_dim, hidden_dim, dataset, epoch)

        signal2model = np.load(CONFIG.GRU_DATA_DIRECTORY + dir_name + '/' + file_tag + ".npz")["signal2model"]
        model = LibphysGRU(signal2model)
        model.load(file_tag, dir_name)
        return model

