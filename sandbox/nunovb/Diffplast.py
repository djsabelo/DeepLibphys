import tensorflow as tf
import math
import os

PATH = "/home/bento/models/"
TYPE = 'plastic'

# Create model
class PlastMLP:
    def __init__(self, n_input, n_hidden_1=32, n_hidden_2=32, n_classes):
        # n_hidden_1: 1st and 3rd layer number of neurons
        # n_hidden_2: 2nd(Latent) layer number of neurons
        # n_input: length of ECG window
        # Store layers weight & bias - try different weights for decoder layers
        lr = 0.001
        self.weights = {
            #'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        # tf.initializers.orthogonal()([n_input, n_hidden_1]),
            'w2': tf.Variable(tf.random_normal([n_hidden_1, n_classes])),
            #'alpha1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            # tf.initializers.orthogonal()([n_input, n_hidden_1]),
            'alpha2': tf.Variable(tf.random_normal([n_hidden_1, n_classes])),
            #'yin1': tf.Variable(tf.random_normal([1, n_hidden_1])),
            # tf.initializers.orthogonal()([n_input, n_hidden_1]),
            'yin2': tf.Variable(tf.random_normal([1, n_classes])),
        }  # , stddev=math.sqrt(6) / math.sqrt(n_input + n_hidden_2 + 1)))}
        # 'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]))}
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_classes])),
            #'b3': tf.Variable(tf.random_normal([n_hidden_1])),
        }
        self.hebb = {
            'h1': tf.Variable(tf.zeros([n_hidden_1])),
            'h2': tf.Variable(tf.zeros([n_classes])),
            # 'b3': tf.Variable(tf.random_normal([n_hidden_1])),
        }
        self.eta = tf.Variable(0.01) # Learning rate

        # tf Graph input
        X = tf.placeholder("float", [None, n_input])

        #self.params = [self.weights, self.biases]  # b_prime + W.T
        if TYPE == 'plastic':
            #self.layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['w1']), self.biases['b1']))
            #self.out = tf.nn.sigmoid(tf.matmul(X, self.weights['w2']))

            self.out = tf.nn.sigmoid(tf.matmul(self.weights['yin2'], self.weights['w2'] + tf.matmul(self.alpha, self.hebb['h2'])) + X)
            self.hebb['h2'] = (1 - self.eta) * self.hebb['h2'] + self.eta * tf.batch_matmul(tf.expand_dims(tf.transpose(self.weights['yin2']), 2), tf.expand_dims(self.out, 1))[0]
            # bmm used to implement outer product with the help of unsqueeze (i.e. added empty dimensions)
        else:
            #self.layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['w1']), self.biases['b1']))
            self.out = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['w2']), self.biases['b2']))

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.squared_difference(self.layer_4, X))
        optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    def fit(self, x, n_epochs=15, learning_rate=0.001, batch_size=64, display_step=1, decay=0.95, save=True, name='1',
            load=False):

        # Y = tf.placeholder("float", [None, n_classes])
        #lr = tf.Variable(learning_rate)
        #self.lr_vector = []
        #self.weight_vec = []
        saver = tf.train.Saver()
        # Hidden fully connected layers
        # - Add sigmoid and try swish
        # - Add noise to x

        # Try to plot costs for different learning rates in order to optimize lr!!!
        self.costs = []



        # Initializing the variables
        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
        if load:
            # saver = tf.train.import_meta_graph(PATH + name + '.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(PATH + '/1/' + './'))  # PATH + './' ))))
            print("Model {} Loaded".format(name))

        # Training cycle
        for epoch in range(n_epochs):
            avg_cost = 0
            total_batch = int(len(x) / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x = x[i * batch_size: i * batch_size + batch_size]
                # Run optimization op (backprop) and cost (to get loss value)
                _, c = self.sess.run([optimizer, cost], feed_dict={X: batch_x})
                # Y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch

            # Update cost vector
            # self.costs.append(avg_cost)
            # self.weight_vec.append(self.sess.run(tf.reduce_mean(self.weights['h2'], axis=0)))

            # Get learning rate
            # lr = tf.cond(tf.logical_and(tf.less(10, len(self.costs)),
            #                             tf.less(tf.reduce_mean(self.costs[-11:-6]),
            #                                     tf.reduce_mean(self.costs[-6:-1]))),
            #              lambda: tf.multiply(lr, decay), lambda: lr)
            # learning_rate = self.sess.run(lr)
            # print("lr:", learning_rate)
            # self.lr_vector.append(learning_rate)

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")

        if save:
            # add subfolder for more than 1 trained model
            save_path = saver.save(self.sess, PATH + name)
            print("Model saved in file: %s" % PATH + name)

    def get_cost_vector(self):
        return self.sess.run(tf.cast(self.costs, dtype=tf.float32))

    def reconstruct(self, x_t):
        # Reconstructs the input signal
        X = tf.placeholder("float", x_t.shape)
        self.layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x_t, self.weights['h1']), self.biases['b1']))
        self.layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(self.layer_1, self.weights['h2']), self.biases['b2']))
        self.layer_3 = tf.nn.sigmoid(
            tf.add(tf.matmul(self.layer_2, tf.transpose(self.weights['h2'])), self.biases['b3']))
        self.layer_4 = tf.matmul(self.layer_3, tf.transpose(self.weights['h1']))
        return self.sess.run(self.layer_4, feed_dict={X: x_t})

    def get_latent(self, x_t):
        # Returns the latent representation of the input signal
        X = tf.placeholder("float", x_t.shape)

        return self.sess.run(self.layer_2, feed_dict={X: x_t})

    def get_weights(self):
        # Returns the latent representation of the input signal
        # X = tf.placeholder("float", x_t.shape)
        return self.weight_vec  # , feed_dict={X: x_t})