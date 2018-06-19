from DeepLibphys.sandbox.nunovb.SRUCell import SRUCell
from DeepLibphys.sandbox.nunovb.indRNNCell import IndRNNCell
import tensorflow as tf
import numpy as np
import time
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import accuracy_score


def binary_activation(x):
    # For hard attention
    cond = tf.less(x, tf.zeros(tf.shape(x)))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))

    return out

timesteps = 500
n_classes = 1
batch_size = 16
samples = 1000
# sequences = np.array([np.array([np.array([np.float32(np.random.randint(0, 15)) for i in range(10)] for i in range(15)])) for i in range(100)]))
sequences = np.array([np.array([np.float32(np.random.random_integers(0, 50)) for i in range(timesteps)]) for i in range(samples)])#.reshape(samples, timesteps, 1)
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
print(sequences.shape)

train = sequences[800:]
test = sequences[-200:]
y_train = labels[800:]
y_test = labels[-200:]
# print(sequences.dtype)
#exit()
#sequences = LabelBinarizer().fit_transform(sequences)
n_units = 32
n_features = 24
#n_heads = 3

x = tf.placeholder(tf.float32, [None, timesteps]) # [timesteps, n_input]
y = tf.placeholder(tf.float32, [None, n_classes])
#input = tf.unstack(x, timesteps, 1)

# Use attention to sample n_heads values from the sequence

# Attention network
l1a = tf.layers.dense(x, n_units, tf.nn.relu)
a = tf.layers.dense(l1a, n_features, tf.nn.softmax)

# Feature Network
l1f = tf.layers.dense(x, n_units, tf.nn.relu)
z = tf.layers.dense(l1f, n_features)

glimpse = z#tf.multiply(a, z)#z

# z = []
# for i in range(n_heads):
#     print(i)
#     l1 = tf.layers.dense(glimpse, n_units, tf.nn.relu)# tf.expand_dims(x[:,tf.cast(samples[0][0], tf.int32)], 1)
#     #l2 = tf.layers.dense(l1, n_classes)
#     z.append(l1)

l2 = tf.layers.dense(glimpse, n_units, tf.nn.relu)

prediction = tf.layers.dense(l2, n_classes) # tf.concat(z, axis=1)

#prediction = tf.nn.softmax(tf.matmul(outputs[:,-1], out_weights) + out_bias)

loss = tf.reduce_mean(tf.squared_difference(prediction,y))#tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

#evaluation = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(evaluation, tf.float32))

sess = tf.Session()
# Initializing the variables
init = tf.global_variables_initializer()
sess.run(init)

total_batch = int(train.shape[0] / batch_size)
#last_sample = np.zeros((x.shape[0], 2, self.n_hidden_1))
# Training cycle
for epoch in range(100):
    avg_cost = 0
    start = time.time()
    # Loop over all batches
    for i in range(total_batch):
        batch_x = train[i * batch_size: i * batch_size + batch_size]
        batch_y = y_train[i * batch_size: i * batch_size + batch_size]
        # Run optimization op (backprop) and cost (to get loss value)
        _, c = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y.reshape(-1,1)})#self.layer_3,self.unpool_1,
        # if epoch % display_step == 0:
        avg_cost += c / total_batch
    print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost), "Time:", time.time() - start, "s")
preds = np.array(sess.run([prediction], feed_dict={x: test}))[0]#.reshape(-1,1)
print(preds[:10] * max_l * max_s)
print(y_test[:10] * max_l * max_s)
print("MSE:",np.mean((preds-y_test)**2))#""Accuracy:", accuracy_score(np.argmax(y_test, 1),np.argmax(preds, 1)))