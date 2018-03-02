import tensorflow as tf
from DeepLibphys.utils.functions.signal2model import Signal2Model
import numpy as np


def to_tensor_flow(name, parameters):
    E, V, U, W, b, c = parameters
    try:
        return tf.convert_to_tensor(eval(name), name=name, dtype=tf.float64)
    except NameError:
        family_name = name[1:]
    return tf.zeros_like(eval(family_name), dtype=tf.float64)


signal2model = Signal2Model("tf_experiment", "TF_Experiment", hidden_dim=4, signal_dim=4, mini_batch_size=2)
coversion_ones = tf.ones((1, signal2model.mini_batch_size), dtype=tf.float64)


def GRU(last_input, gru_params):
    i, s_prev = gru_params
    i1, i2, i3 = i * 3 + 0, i * 3 + 1, i * 3 + 2
    Hd = tf.shape(b)[1]

    b1 = tf.matmul(tf.reshape(tf.gather(b, i1), (Hd, 1)), coversion_ones)
    b2 = tf.matmul(tf.reshape(tf.gather(b, i2), (Hd, 1)), coversion_ones)
    b3 = tf.matmul(tf.reshape(tf.gather(b, i3), (Hd, 1)), coversion_ones)

    w1, w2, w3 = tf.reshape(tf.gather(W, i1), (Hd, Hd)), \
                 tf.reshape(tf.gather(W, i2), (Hd, Hd)), \
                 tf.reshape(tf.gather(W, i3), (Hd, Hd))

    u1, u2, u3 = tf.reshape(tf.gather(U, i1), (Hd, Hd)), \
                 tf.reshape(tf.gather(U, i2), (Hd, Hd)), \
                 tf.reshape(tf.gather(U, i3), (Hd, Hd))

    z = tf.nn.sigmoid(tf.matmul(u1, last_input) + tf.matmul(w1, s_prev) + b1)
    r = tf.nn.sigmoid(tf.matmul(u2, last_input) + tf.matmul(w2, s_prev) + b2)

    value = tf.matmul(u3, last_input) + tf.matmul(w3, s_prev * r) + b3
    s_candidate = tf.nn.tanh(value)

    return ((tf.ones_like(z) - z) * s_candidate) + (z * s_prev)


if __name__ == "__main__":
    signal2model = Signal2Model("tf_experiment", "TF_Experiment", hidden_dim=4, signal_dim=4, mini_batch_size=2)

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
    x = tf.Variable('x')
    y = tf.Variable('y')
    parameters_names = ["E", "V", "U", "W", "b", "c"]  # parameter names
    parameters_m_names = ["m" + par_name
                               for par_name in parameters_names]  # SGD parameter names (for RMSProp)
    parameters = [E, U, W, V, b, c]
    variables_dict = {par_name: to_tensor_flow(par_name, parameters)
                           for par_name in parameters_names + parameters_m_names}

    intial_values = tf.zeros((signal2model.hidden_dim, signal2model.mini_batch_size), dtype=tf.float64)


gru_indexes = np.reshape(np.arange(signal2model.n_grus, dtype=np.int32), (3, 1))
Hd = signal2model.hidden_dim


def GRUnn(out_prev, x_t):
    s_prev, o_prev = out_prev

    # Embedding layer
    x_e = tf.transpose(tf.gather(E, x_t))

    s = tf.scan(GRU, (gru_indexes, s_prev), initializer=x_e, parallel_iterations=1)
    linear_node = tf.matmul(V, s[-1]) + tf.matmul(tf.reshape(c, (signal2model.signal_dim, 1)), coversion_ones)

    o_t = tf.nn.softmax(linear_node)

    return [s, o_t]


def feed_forward_predict(X):
    initial_s = np.zeros((signal2model.n_grus, signal2model.hidden_dim, signal2model.mini_batch_size), dtype=np.float64)
    initial_x = tf.placeholder(dtype=tf.float64, shape=[None, signal2model.mini_batch_size])

    out = tf.scan(GRUnn, X, initializer=[initial_s, initial_x], parallel_iterations=1, infer_shape=False)
    return tf.argmax(out[-1], axis=1)


def calculate_predictions(X):
    x_c = np.reshape(X, (int(np.shape(X)[1] / signal2model.mini_batch_size), np.shape(X)[0], signal2model.mini_batch_size))
    return tf.map_fn(feed_forward_predict, x_c)


def calculate_mse(X, Y):
    return (tf.reshape(calculate_predictions(X), np.shape(Y)) - Y) ** 2


def calculate_mse_vector_loss(X, Y):
    return tf.reduce_mean(calculate_mse(X, Y), axis=0)


def calculate_mse_loss(X, Y):
    return tf.reduce_mean(calculate_mse(X, Y))


X = np.ones((20, 6), dtype=np.int64)
Y = np.ones((20, 6), dtype=np.int64)

session = tf.Session()
result = session.run(calculate_mse_vector_loss(X, Y))

print(result)


