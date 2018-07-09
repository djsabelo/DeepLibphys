import tensorflow as tf
import numpy as np
import time
import functools
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score


def binary_activation(x):
    # For hard attention
    cond = tf.less(x, tf.zeros(tf.shape(x)))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return out

def gen_masked_sequences(min_length, n_sequences, sample, max_length=None):
    """ Generates two-dimensional sequences and masks of length randomly chosen
    between ``[min_length, max_length]`` where one dimension is randomly
    sampled using ``sample`` and the other dimension has -1 at the start and
    end of the sequence and has a 1 in the first ten sequence steps and another
    1 before ``min_length/2`` and is 0 otherwise.
    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    n_sequences : int
        Number of sequences to generate.
    sample : function
        Function used to randomly sample values; should accept a keyword
        argument "size" which determines the sampled shape.
    max_length : int or None
        Maximum sequence length.  If supplied as `None`,
        ``int(np.ceil(1.1*min_length))`` will be used.
    Returns
    -------
    X : np.ndarray
        Input to the network, of shape
        ``(n_sequences, 1.1*min_length, 2)``, where the last
        dimension corresponds to the two sequences described above.
    mask : np.ndarray
        A binary matrix of shape ``(n_sequences, 1.1*min_length)``
        where ``mask[i, j] = 1`` when ``j <= (length of sequence i)``
        and ``mask[i, j] = 0`` when ``j > (length of sequence i)``.
    """
    if max_length is None:
        # Compute the global maximum sequence length
        max_length = int(np.ceil(1.1*min_length))
    # When min_length and max_length is the same, make all lengths the same
    if min_length == max_length:
        lengths = min_length*np.ones(n_sequences, dtype=int)
    # Otherwise, randomly choose lengths from [min_length, max_length]
    else:
        lengths = np.random.randint(min_length, max_length, n_sequences)
    # Construct a mask by tiling arange and comparing it to lengths
    mask = (np.tile(np.arange(max_length), (n_sequences, 1)) <
            np.tile(lengths, (max_length, 1)).T)
    # Sample the noise dimension
    noise_dim = sample(size=(n_sequences, max_length, 1))
    # Mask out all entries past the end of each sequence
    noise_dim *= mask[:, :, np.newaxis]
    # Initialize mask dimension to all zeros
    mask_dim = np.zeros((n_sequences, max_length, 1))
    # First entry of each sequence in mask dimension is -1
    mask_dim[:, 0] = -1
    # End of sequence in mask dimension is -1
    mask_dim[np.arange(n_sequences), lengths - 1] = -1
    # First add index is always between 1 and 10
    N_1 = np.random.randint(1, 10, n_sequences)
    # Second add index is always between 1 and length/2
    # Using max_length instead of length[n] is a WLOG hack
    N_2 = np.random.choice(range(1, int(max_length/2 - 1)), n_sequences)
    # If N_1 = N_2 for any sequences, add 1 to avoid
    N_2[N_2 == N_1] = N_2[N_2 == N_1] + 1
    # Set the add indices to 1
    mask_dim[np.arange(n_sequences), N_1] = 1
    mask_dim[np.arange(n_sequences), N_2] = 1
    # Concatenate noise and mask dimensions to create data
    X = np.concatenate([noise_dim, mask_dim], axis=-1)
    return X, mask

def add(min_length, n_sequences, max_length=None):
    """ Generate sequences and target values for the "add" task, as described in
    [1]_ section 5.4.  Sequences are two dimensional where the first dimension
    are values sampled uniformly at random from [-1, 1] and the second
    dimension is either -1, 0, or 1: At the first and last steps, it is
    -1; at one of the first ten steps (``N_1``) it is 1; and at a step between
    0 and ``.5*min_length`` (``N_2``) it is
    also 1.  The goal is to predict ``0.5 + (X_1 + X_2)/4.0`` where ``X_1`` and
    ``X_2`` are the values of the first dimension at ``N_1`` and ``N_2``
    respectively.  For example, the target for the following sequence
    ``| 0.5 | -0.7 | 0.3 | 0.1 | -0.2 | ... | -0.5 | 0.9 | ... | 0.8 | 0.2 |
      | -1  |   0  |  1  |  0  |   0  |     |   0  |  1  |     |  0  | -1  |``
    would be ``.5 + (.3 + .9)/4 = .8``.  All generated sequences will be of
    length ``max_length``; the returned variable ``mask``
    can be used to determine which entries are in each sequence.
    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    n_sequences : int
        Number of sequences to generate.
    max_length : int or None
        Maximum sequence length.  If supplied as `None`,
        ``int(np.ceil(1.1*min_length))`` will be used.
    Returns
    -------
    X : np.ndarray
        Input to the network, of shape
        ``(n_sequences, 1.1*min_length, 2)``, where the last
        dimension corresponds to the two sequences described above.
    y : np.ndarray
        Correct output for each sample, shape ``(n_sequences,)``.
    mask : np.ndarray
        A binary matrix of shape ``(n_sequences, 1.1*min_length)``
        where ``mask[i, j] = 1`` when ``j <= (length of sequence i)``
        and ``mask[i, j] = 0`` when ``j > (length of sequence i)``.
    References
    ----------
    .. [1] Sepp Hochreiter and Jürgen Schmidhuber. "Long short-term memory."
    Neural computation 9.8 (1997): 1735-1780.
    """
    # Get sequences
    X, mask = gen_masked_sequences(
        min_length, n_sequences,
        functools.partial(np.random.uniform, high=1., low=-1.), max_length)
    # Sum the entries in the third dimension where the second is 1
    y = np.sum((X[:, :, 0]*(X[:, :, 1] == 1)), axis=1)
    # Normalize targets to the range [0, 1]
    y = .5 + y/4.
    return X, y, mask

def multiply(min_length, n_sequences, max_length=None):
    """ Generate sequences and target values for the "multiply" task, as
    described in [1]_ section 5.5.  Sequences are two dimensional where the
    first dimension are values sampled uniformly at random from [0, 1] and the
    second dimension is either -1, 0, or 1: At the first and last steps, it is
    -1; at one of the first ten steps (``N_1``) it is 1; and at a step between
    0 and ``.5*min_length`` (``N_2``) it is also 1.  The goal is to predict
    ``X_1*X_2`` where ``X_1`` and ``X_2`` are the values of the first dimension
    at ``N_1`` and ``N_2`` respectively.  For example, the target for the
    following sequence
    ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
      | -1  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  | -1  |``
    would be ``.3*.9 = .27``.  All generated sequences will be of
    length ``max_length``; the returned variable ``mask``
    can be used to determine which entries are in each sequence.
    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    n_sequences : int
        Number of sequences to generate.
    max_length : int or None
        Maximum sequence length.  If supplied as `None`,
        ``int(np.ceil(1.1*min_length))`` will be used.
    Returns
    -------
    X : np.ndarray
        Input to the network, of shape
        ``(n_sequences, 1.1*min_length, 2)``, where the last
        dimension corresponds to the two sequences described above.
    y : np.ndarray
        Correct output for each sample, shape ``(n_sequences,)``.
    mask : np.ndarray
        A binary matrix of shape ``(n_sequences, 1.1*min_length)``
        where ``mask[i, j] = 1`` when ``j <= (length of sequence i)``
        and ``mask[i, j] = 0`` when ``j > (length of sequence i)``.
    References
    ----------
    .. [1] Sepp Hochreiter and Jürgen Schmidhuber. "Long short-term memory."
    Neural computation 9.8 (1997): 1735-1780.
    """
    # Get sequences
    X, mask = gen_masked_sequences(
        min_length, n_sequences,
        functools.partial(np.random.uniform, high=1., low=0.), max_length)
    # Multiply the entries in the third dimension where the second is 1
    y = np.prod((X[:, :, 0]**(X[:, :, 1] == 1)), axis=1)
    return X, y, mask



# sequences = np.array([np.array([np.array([np.float32(np.random.randint(0, 15)) for i in range(10)] for i in range(15)])) for i in range(100)]))
'''sequences = np.array([np.array([np.float32(np.random.random_integers(0, 50)) for i in range(timesteps)]) for i in range(samples)])#.reshape(samples, timesteps, 1)
#print(sequences[:5])
#exit()
max_s = np.max(sequences)
sequences /= max_s
#    for i in range(N_Windows):
#print(np.array([np.sum(seq[:3]) for seq in sequences]).shape)
labels = np.array([np.sum(seq[:timesteps // 10:4]) for seq in sequences])#LabelBinarizer().fit_transform(np.array([np.sum(seq[3:5]) for seq in sequences]))
max_l = np.max(labels)
labels /= max_l

# sequences = np.array([np.array(LabelBinarizer().fit_transform([np.float32(np.random.randint(0, 15)) for i in range(150)])) for i in range(100)])
#labels = LabelBinarizer().fit_transform([np.mean(seq[:3]) for seq in sequences])
print(sequences.shape)'''


#timesteps = 20
total_time = 600
n_classes = 1
batch_size = 16
samples = 1000
n_features = 2

X, y, mask = multiply(total_time, 1000, total_time)
print(X.shape)
print(y.shape)

train = X[800:]
test = X[-200:].reshape(-1,2)
y_train = y[800:]
y_test = y[-200:]
# print(sequences.dtype)
#exit()
#sequences = LabelBinarizer().fit_transform(sequences)
n_units = 32
#n_features = 10
#n_heads = 3

x = tf.placeholder(tf.float32, [None, n_features]) # [timesteps, n_input]
y = tf.placeholder(tf.float32, [None, n_classes])
#input = tf.unstack(x, timesteps, 1)

# Use attention to sample n_heads values from the sequence

# Attention network
#X = x#tf.reshape(x, [-1,2])
#l1a = tf.layers.dense(X, n_units, tf.nn.relu)
#a = tf.layers.dense(l1a, n_features, tf.nn.softmax)
#a = tf.reshape(a, [-1, total_time, n_features])
# Feature Network
#z = tf.layers.dense(X, n_units, tf.nn.relu)
#z = tf.layers.dense(z, n_features, tf.nn.relu)
X = tf.reshape(x, [-1, total_time, n_features])

# No residual block
# l1 = tf.layers.conv1d(X, n_units, 8, dilation_rate=1)
# l2 = tf.layers.conv1d(l1, n_units, 4, dilation_rate=2)
# l3 = tf.layers.conv1d(l2, n_units, 2, dilation_rate=3)
# l3 = tf.layers.conv1d(l3, n_units, 2, dilation_rate=4)
# l5 = tf.layers.conv1d(l4, n_units, 1, dilation_rate=5)

# Residual blocks
with tf.variable_scope('Res_conv1d'):
    l1 = tf.layers.conv1d(X, n_units, 4, dilation_rate=1)
    l1 = tf.nn.relu(tf.layers.batch_normalization(l1))#tf.nn.dropout(tf.nn.relu(tf.layers.batch_normalization(l1)), 0.99)
    l2 = tf.layers.conv1d(l1, n_units, 4, dilation_rate=2)
    l2 = tf.nn.relu(tf.layers.batch_normalization(l2))#tf.nn.dropout(tf.nn.relu(tf.layers.batch_normalization(l2)), 0.99)
    # Residual connections every two layers
    # 1x1 conv
    # l_1x1 = tf.layers.conv1d(X, n_units, 1, dilation_rate=1)
    l3 = tf.layers.conv1d(l2, n_units, 3, dilation_rate=4) # tf.concat([l2, l_1x1], 1)
    l3 = tf.nn.relu(tf.layers.batch_normalization(l3))#tf.nn.dropout(tf.nn.relu(tf.layers.batch_normalization(l3)), 0.99)
    l3 = tf.layers.conv1d(l3, n_units, 2, dilation_rate=8)  # tf.concat([l2, l_1x1], 1)
    l3 = tf.nn.relu(tf.layers.batch_normalization(l3))
    # l4 = tf.layers.conv1d(l3, n_units, 2, dilation_rate=4, activation=tf.nn.relu)
    # l5 = tf.layers.conv1d(l4, n_units, 1, dilation_rate=5, activation=tf.nn.relu)

# Mean
#glimpse = tf.reduce_mean(l4,1)#tf.multiply(a, z)#tf.reduce_sum(tf.multiply(a, z),1)

# Attention
#a = tf.nn.softmax(tf.layers.dense(z, n_units, tf.nn.relu),axis=1)
#glimpse = tf.reduce_sum(tf.multiply(a, z),1)#tf.expand_dims(a, 2)


d = tf.layers.flatten(tf.layers.dense(l3, n_units, tf.nn.relu))

prediction = tf.layers.dense(d, n_classes) # tf.concat(z, axis=1)

#prediction = tf.nn.softmax(tf.matmul(outputs[:,-1], out_weights) + out_bias)

loss = tf.reduce_mean(tf.squared_difference(prediction,y))#tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0007).minimize(loss)

#evaluation = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(evaluation, tf.float32))

sess = tf.Session()
# Initializing the variables
init = tf.global_variables_initializer()
sess.run(init)

total_batch = int(train.shape[0] / batch_size)
#last_sample = np.zeros((x.shape[0], 2, self.n_hidden_1))
# Training cycle
for epoch in range(50):
    avg_cost = 0
    start = time.time()
    # Loop over all batches
    for i in range(total_batch):
        batch_x = train[i * batch_size: i * batch_size + batch_size].reshape(-1,2)
        batch_y = y_train[i * batch_size: i * batch_size + batch_size]#np.repeat(y_train[i * batch_size: i * batch_size + batch_size], len(train)/timesteps)
        # Run optimization op (backprop) and cost (to get loss value)
        _, c = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y.reshape(-1,1)})#self.layer_3,self.unpool_1,
        # if epoch % display_step == 0:
        avg_cost += c / total_batch
    print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost), "Time:", time.time() - start, "s")
preds = np.array(sess.run([prediction], feed_dict={x: test}))[0].reshape(-1)
print(preds[:10])
print(y_test[:10])
print("Accuracy:",np.mean(np.abs(preds - y_test) < .04))#""Accuracy:", accuracy_score(np.argmax(y_test, 1),np.argmax(preds, 1)))
#print("Accuracy:", accuracy_score(np.argmax(y_test, 1),np.argmax(preds, 1)))
