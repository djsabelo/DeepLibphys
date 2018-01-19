import tensorflow as tf
import numpy as np
import math
import os

PATH = "/home/bento/models/"

# Create model
class Autoencoder:
    def __init__(self, n_input=1024, n_hidden_1=256, n_hidden_2=2):
        # n_hidden_1: 1st and 3rd layer number of neurons
        # n_hidden_2: 2nd(Latent) layer number of neurons
        # n_input: length of ECG window
        # Store layers weight & bias - try different weights for decoder layers
        self.n_hidden_1 = n_hidden_1
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])), #tf.initializers.orthogonal()([n_input, n_hidden_1]),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))}#, stddev=math.sqrt(6) / math.sqrt(n_input + n_hidden_2 + 1)))}
            #'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]))}
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([n_hidden_1])),
        }
        
        self.params = [self.weights, self.biases] #b_prime + W.T

    def fit(self, x, learning_rate=0.001, batch_size=64, display_step=1, decay=0.95, save=True, name='1',
            load=False):
        # tf Graph input
        X = tf.placeholder("float", [None, x.shape[1]])
        # Y = tf.placeholder("float", [None, n_classes])
        w_list = np.arange(-1, 1, 0.1)
        syn1 = np.array([np.random.uniform(w_list[i]-0.1,w_list[i]+0.1,self.n_hidden_1) for i in range(20)])
        syn2 = np.array([np.random.uniform(w_list[i] - 0.1, w_list[i] + 0.1, self.n_hidden_1) for i in range(20)])
        #syn1 = [np.float32(item) for item in syn1]
        #syn2 = [np.float32(item) for item in syn2]
        weights = []
        # weights = np.vstack([syn1[0],syn2[0]]).T
        # print(weights.shape)
        # exit()
        # Try to plot costs for different learning rates in order to optimize lr!!!
        self.costs = []
        for i in range(20):
            for j in range(20):
                #print(tf.constant(np.vstack([syn1[i],syn2[j]]).T).dtype)
                # Hidden fully connected layers
                self.layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['h1']), self.biases['b1']))
                self.layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(self.layer_1, tf.convert_to_tensor(np.vstack([syn1[i],syn2[j]]).T, tf.float32)), self.biases['b2']))
                self.layer_3 = tf.nn.sigmoid(tf.matmul(self.layer_2, tf.convert_to_tensor(np.vstack([syn1[i],syn2[j]]), tf.float32)))
                self.layer_4 = tf.matmul(self.layer_3, tf.transpose(self.weights['h1']))  # , self.biases['b1'])


                # Define loss and optimizer
                cost = tf.reduce_mean(tf.squared_difference(self.layer_4, x))

                # Initializing the variables
                init = tf.global_variables_initializer()

                self.sess = tf.Session()
                self.sess.run(init)

                print(self.sess.run(tf.convert_to_tensor(np.vstack([syn1[i], syn2[j]]).T, tf.float32)))
                _, cost = self.sess.run(cost, feed_dict={X: x})

                # Update cost vector
                self.costs.append(cost)
                weights.append(np.mean(np.vstack([syn1[i], syn2[j]])))

        return np.array(self.costs), np.array(weights)
        #print("Optimization Finished!")


    def get_cost_vector(self):
        return self.sess.run(tf.cast(self.costs, dtype=tf.float32))
    
    def reconstruct(self, x_t):
        # Reconstructs the input signal
        X = tf.placeholder("float", x_t.shape)
        return self.sess.run(self.layer_4, feed_dict={X: x_t})

    def get_latent(self, x_t):
        # Returns the latent representation of the input signal
        X = tf.placeholder("float", x_t.shape)
        return self.sess.run(self.layer_2, feed_dict={X: x_t})

    def get_weights(self):
        # Returns the latent representation of the input signal
        #X = tf.placeholder("float", x_t.shape)
        return self.weight_vec#, feed_dict={X: x_t})