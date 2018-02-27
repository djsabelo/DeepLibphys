import tensorflow as tf
import math
import os
import numpy as np

PATH = "/home/bento/models/"

# Create model
class Autoencoder:
    def __init__(self, n_input=1024, n_hidden_1=128, n_hidden_2=16):
        # n_hidden_1: 1st and 3rd layer number of neurons
        # n_hidden_2: 2nd(Latent) layer number of neurons
        # n_input: length of ECG window
        # Store layers weight & bias - try different weights for decoder layers
        self.n_input = n_input
        self.n_hidden_1 = n_hidden_1
        #self.n_hidden_2 = n_hidden_2
        '''self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])), #tf.initializers.orthogonal()([n_input, n_hidden_1]),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))}#, stddev=math.sqrt(6) / math.sqrt(n_input + n_hidden_2 + 1)))}
            #'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]))}
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([n_hidden_1])),
        }'''
        
        #self.params = [self.weights, self.biases] #b_prime + W.T

    def fit(self, x, n_epochs=15, learning_rate=0.001, batch_size=32, display_step=1, save=True, name='1', load=False):
        # tf Graph input
        self.graph = tf.Graph()
        with self.graph.as_default():
            X = tf.placeholder("float", [None,x.shape[1],1])
        #Y = tf.placeholder("float", [None, n_classes])


        # Hidden fully connected layers
        # - Add sigmoid and try swish
        # - Add noise to x
        # self.layer_1 = tf.layers.conv1d(x, self.n_hidden_1, kernel_size=16)
        # self.layer_2 = tf.layers.conv1d(self.layer_1, tf.transpose(self.n_hidden_1), kernel_size=16) # What to transpose?
        #self.layer_3 =
        #self.layer_4 =
        #x = tf.convert_to_tensor(x, np.float32)
            self.build_model(x)

        # self.layer_1 = tf.layers.conv1d(tf.convert_to_tensor(x, np.float32), self.n_hidden_1, kernel_size=16,
        #                                 padding='same')
        # self.layer_2 = tf.layers.conv1d(self.layer_1, x.shape[1], kernel_size=16, padding='same')

            saver = tf.train.Saver()
            # Try to plot costs for different learning rates in order to optimize lr!!!
            self.costs = []

            # Define loss and optimizer
            cost = tf.reduce_mean(tf.squared_difference(self.layer_2, x))
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

            # Initializing the variables
            init = tf.global_variables_initializer()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)
        if load:
            #saver = tf.train.import_meta_graph(PATH + name + '.meta')
            #saver.restore(self.sess, tf.train.latest_checkpoint(PATH + './'))#PATH + './' ))))
            saver.restore(self.sess, PATH + name + '.ckpt')
            print("Model {} Loaded".format(name))
        
        # Training cycle
        for epoch in range(n_epochs):
            avg_cost = 0
            total_batch = int(x.shape[0]/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x = x[i * batch_size: i * batch_size + batch_size]
                # Run optimization op (backprop) and cost (to get loss value)
                _, c = self.sess.run([optimizer, cost], feed_dict={X: batch_x})
                                                                #Y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch

            # Update cost vector
            self.costs.append(avg_cost)
            
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")

        if save:
            # add subfolder for more than 1 trained model
            #saver.save(self.sess, PATH + name)
            saver.save(self.sess, PATH + name + '.ckpt')
            print("Model saved in file: %s" % PATH + name)

    def build_model(self, x):
        # with self.graph.as_default():
        self.layer_1 = tf.layers.conv1d(tf.convert_to_tensor(x, np.float32), self.n_hidden_1, kernel_size=64, padding='same')
        self.layer_2 = tf.layers.conv1d(self.layer_1, x.shape[1], kernel_size=64, padding='same')

    def get_cost_vector(self):
        return self.sess.run(tf.cast(self.costs, dtype=tf.float32))
    
    def reconstruct(self, x_t):
        # Reconstructs the input signal
        with self.graph.as_default():
            X = tf.placeholder("float", [None,x_t.shape[1],1])
            self.build_model(x_t)
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
        return self.sess.run(self.layer_2, feed_dict={X: x_t})
        
    def get_latent(self, x_t):
        # Returns the latent representation of the input signal
        X = tf.placeholder("float", x_t.shape)
        return self.sess.run(self.layer_2, feed_dict={X: x_t})
    
