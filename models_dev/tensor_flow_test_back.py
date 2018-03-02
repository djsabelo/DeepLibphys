import tensorflow as tf
import numpy as np
# Creates a graph.
# Creates a graph.
# with tf.device('/cpu:0'):
#   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#   b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))



# config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True )#choose automaticaly the device
# config.gpu_options.allow_growth = True # allow the process to grow memory
# config.gpu_options.per_process_gpu_memory_fraction = 0.4 # choose a percentage of GPU


session = tf.Session(config=tf.ConfigProto(log_device_placement=True ))#config=config)

# # Creates a graph.
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=config)
# # Runs the op.
# print(sess.run(c))

# Creates a graph.
c = []
for d in ['/device:GPU:0', '/device:GPU:1']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)

def start(x_in, y_in, signal2model):
    E = np.random.uniform(-np.sqrt(1. / signal2model.signal_dim), np.sqrt(1. / signal2model.signal_dim),
                          (signal2model.hidden_dim, signal2model.signal_dim))
    U = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                          (signal2model.hidden_dim, signal2model.hidden_dim))
    W = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                          (signal2model.hidden_dim, signal2model.hidden_dim))
    V = np.random.uniform(-np.sqrt(1. / signal2model.hidden_dim), np.sqrt(1. / signal2model.hidden_dim),
                          (signal2model.signal_dim, signal2model.hidden_dim))

    b = np.zeros((signal2model.hidden_dim))
    c = np.zeros((signal2model.signal_dim))


    E = tf.Variable(E, name="E")
    U = tf.Variable(U, name="U")
    W = tf.Variable(W, name="W")
    b = tf.Variable(b, name="b")
    c = tf.Variable(c, name="c")

    x = tf.contant(x_in)
    y = tf.contant(y_in)

    var = tf.global_variables_initializer()


    coversion_ones = tf.ones((signal2model.mini_batch_size, 1))

    def GRU(i, U, W, b, x_0, s_previous):
        U_copy, W_copy = U, W
        b1 = tf.matmul(coversion_ones, b[i * 3, :])
        b2 = tf.matmul(coversion_ones, b[i * 3 + 1, :])
        b3 = tf.matmul(coversion_ones, b[i * 3 + 2, :])

        z = tf.sigmoid(tf.add(tf.add(tf.matmul(U[i * 3 + 0], x_0), tf.matmul(W[i * 3 + 0],s_previous)), b1))
        r = tf.sigmoid(tf.add(tf.add(tf.matmul(U[i * 3 + 1], x_0), tf.matmul(W[i * 3 + 1], s_previous)), b2))
        s_candidate = tf.tanh(tf.add(tf.add(tf.matmul(U_copy[i * 3 + 2], x_0),
                                            tf.matmul(W[i * 3 + 2], tf.matmul(s_previous, r)), b3)))

        return tf.add(tf.matmul(tf.ones_like(z) - z, s_candidate), tf.matmul(z, s_previous))


    def forward_prop_step(x_t, s_prev):
        # Embedding layer
        x_0 = E[:, x_t[:, 0]]
        s = tf.zeros_like(s_prev)

        # GRU BLOCK 1 [1 vs 1] #############
        # GRU Layer 1
        # print(x_e0)
        # print(s_prev)

        s[0] = GRU(0, U, W, b, x_0, s_prev[0])
        s[1] = GRU(1, U, W, b, s[0], s_prev[1])
        s[2] = GRU(2, U, W, b, s[1], s_prev[2])


        # Final output calculation
        # FIRST DIMENSION:
        o_t =tf.sparse_softmax()

            T.stack(T.nnet.softmax((V[0].dot(s_[:, 0, 1]) + conversion_ones.dot(c[0])))[0],
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

    print(session.run(c))