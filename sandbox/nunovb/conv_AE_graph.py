import tensorflow as tf
import time
import math
import matplotlib.pyplot as plt
import os
import numpy as np

PATH = "/home/bento/models/"

# Create model
class Autoencoder:
    def __init__(self, n_input=1024, n_hidden_1=4, n_hidden_2=2, batch_size=64):
        # n_hidden_1: 1st and 3rd layer number of neurons
        # n_hidden_2: 2nd(Latent) layer number of neurons
        # n_input: length of ECG window
        # Store layers weight & bias - try different weights for decoder layers
        self.n_input = n_input
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        #self.batch_size = batch_size
        '''self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])), #tf.initializers.orthogonal()([n_input, n_hidden_1]),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))}#, stddev=math.sqrt(6) / math.sqrt(n_input + n_hidden_2 + 1)))}
            #'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]))}
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([n_hidden_1])),
        }'''

        self.X = tf.placeholder("float", [None, n_input, 1])
        # self.last_sample = tf.placeholder("float", [None, 2, self.n_hidden_1])
        #self.last_sample = tf.Variable(tf.zeros((batch_size, 2, self.n_hidden_1)))

        self.build_model()
        self.predictions = self.layer_4

        #self.params = [self.weights, self.biases] #b_prime + W.T

    def fit(self, x, n_epochs=15, learning_rate=0.001, batch_size=32, decay=0.95, display_step=1, save=True, name='1', load=False):
        # tf Graph input
        #self.graph = tf.Graph()
        #with self.graph.as_default():

        #Y = tf.placeholder("float", [None, n_classes])


        # Hidden fully connected layers
        # - Add sigmoid and try swish
        # - Add noise to x
        # self.layer_1 = tf.layers.conv1d(x, self.n_hidden_1, kernel_size=16)
        # self.layer_2 = tf.layers.conv1d(self.layer_1, tf.transpose(self.n_hidden_1), kernel_size=16) # What to transpose?
        #self.layer_3 =
        #self.layer_4 =
        #x = tf.convert_to_tensor(x, np.float32)

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.squared_difference(self.predictions, self.X))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    # self.layer_1 = tf.layers.conv1d(tf.convert_to_tensor(x, np.float32), self.n_hidden_1, kernel_size=16,
    #                                 padding='same')
    # self.layer_2 = tf.layers.conv1d(self.layer_1, x.shape[1], kernel_size=16, padding='same')

        saver = tf.train.Saver()
        # Try to plot costs for different learning rates in order to optimize lr!!!
        self.costs = []
        self.tmpcosts = []


        self.sess = tf.Session()
        # Initializing the variables
        init = tf.global_variables_initializer()


        self.sess.run(init)
        if load:
            #saver = tf.train.import_meta_graph(PATH + name + '.meta')
            #saver.restore(self.sess, tf.train.latest_checkpoint(PATH + './'))#PATH + './' ))))
            saver.restore(self.sess, PATH + name + '.ckpt')
            print("Model {} Loaded".format(name))

        total_batch = int(x.shape[0] / batch_size)
        #last_sample = np.zeros((x.shape[0], 2, self.n_hidden_1))
        # Training cycle
        for epoch in range(n_epochs):
            avg_cost = 0
            start = time.time()
            # Loop over all batches
            for i in range(total_batch):
                batch_x = x[i * batch_size: i * batch_size + batch_size]
                # last_sample = last_sample[i * batch_size: i * batch_size + batch_size]
                # Run optimization op (backprop) and cost (to get loss value)
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.X: batch_x})#self.layer_3,self.unpool_1,
                                                                #Y: batch_y})

                # print("l3",l3[0])#WHY???
                # # l3[[[-0.81094015 - 0.30336073 - 0.8713582... - 0.82848024  1.1948334
                # #      - 0.8535064]]
                # #
                # # [[-0.76295 - 0.13263944 - 0.98478407... - 0.78219986  1.3200742
                # #   - 0.90355676]]
                # print(u1[0])
                # plt.subplot(211)
                # plt.title("Layer 3")
                # plt.ylabel('Normalized Voltage')
                # plt.xlabel('Samples')
                # plt.plot(l3[0])
                # plt.subplot(212)
                # plt.title("UP 1")
                # plt.ylabel('Normalized Voltage')
                # plt.xlabel('Samples')
                # plt.plot(u1[0])
                # plt.show()
                #exit()
                # Compute average loss
                avg_cost += c / total_batch

            # Update cost vector
            self.costs.append(avg_cost)
            self.tmpcosts.append(avg_cost)
            # Adaptive learning rate
            if len(self.tmpcosts) > 10 and self.tmpcosts[-10:-5] < self.tmpcosts[-5:]:
                learning_rate *= decay
                print("Lr is", learning_rate)
                self.tmpcosts = []

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost), "Time:", time.time() - start, "s")
        print("Optimization Finished!")

        if save and self.costs[-(n_epochs//2)] > self.costs[-1]:
            # add subfolder for more than 1 trained model
            #saver.save(self.sess, PATH + name)
            saver.save(self.sess, PATH + name + '.ckpt')
            print("Model saved in file: %s" % PATH + name)

    def build_model(self):
        # with self.graph.as_default():
        #print(tf.convert_to_tensor(x, np.float32))
        #regularizer = tf.contrib.layers.l2_regularizer(scale=0.01) # Best: 737
        self.layer_1 = tf.layers.conv1d(self.X, self.n_hidden_1, kernel_size=7, # try 80,50,20
                                        padding='same')#, kernel_regularizer=regularizer)
        # print(self.layer_1)
        self.pool_1 = tf.layers.max_pooling1d(self.layer_1, 2, strides=2, padding='same')
        # print(self.pool_1)
        self.layer_2 = tf.layers.conv1d(self.pool_1, self.n_hidden_2, kernel_size=3, padding='same')#, kernel_regularizer=regularizer)
        # print(self.layer_2)
        self.layer_3 = tf.layers.conv1d(self.layer_2, self.n_hidden_1, kernel_size=3, padding='same')#, kernel_regularizer=regularizer)

        #print(self.layer_3)
        # self.unpool_1 = tf.Variable(np.zeros((64, 1024, self.n_hidden_1)), dtype=tf.float32)
        # x = tf.map_fn(self.interpolate, tf.range(0, tf.shape(self.layer_3)[1]), dtype=tf.float32)
        # self.unpool_1 = tf.reshape(tf.map_fn(self.interpolate,\
        #     tf.range(0, tf.shape(self.layer_3)[1]), dtype=tf.float32), shape=(-1,1024,self.n_hidden_1))#,(2 * self.n_input) - 2))

        #sample = tf.reshape(self.layer_3[:,-2,:], shape=(-1,2,self.n_hidden_1)) # Get sample from last iteration!
        self.unpool_1 = tf.transpose(tf.reshape(tf.map_fn(self.interpolate,\
            tf.range(0, tf.shape(self.layer_3)[1]), dtype=tf.float32), (1024, -1, 4)), perm=[1, 0, 2])
        #     self.layer_3, dtype=tf.float32), shape=(-1,1024,self.n_hidden_1))# tf.concat([self.last_sample, dsf], axis=1) # axis???
        # self.last_sample = self.unpool_1[:,-2,:]
        #interp = tf.map_fn(lambda x: (x[0] + x[1]) / 2, elems=(self.layer_3[:,:-1,:], self.layer_3[:,1:,:]))#, self.layer_3))
        #self.unpool_1 = tf.concat([self.layer_3, dsf], axis=1)
        # tf.stack(self.layer_3, [(self.layer_3[:,:,s] + self.layer_3[:,:,s - 1]) / 2 for s in range(self.layer_3.get_shape().as_list()[2])])
            # tf.reshape(tf.map_fn(self.interpolate,\
            # tf.range(1, tf.shape(self.layer_3)[2]), dtype=tf.float32), shape=(-1,1,512))
            # Copy list and apply lambda

        #self.unpool_1 = tf.keras.layers.UpSampling1D(2)(self.layer_3)#(tf.transpose(self.layer_3, perm=[0,2,1]))# Change to interpolation!!!
        # print(self.unpool_1)
        #self.unpool_1.
        self.layer_4 = tf.layers.conv1d(self.unpool_1, 1, kernel_size=7, padding='same')#, kernel_regularizer=regularizer)

        #tf.transpose(self.unpool_1, perm=[0, 2, 1])
        #print(self.layer_2)

    def get_cost_vector(self):
        return self.sess.run(tf.cast(self.costs, dtype=tf.float32))

    def reconstruct(self, x_t):
        # Reconstructs the input signal
        #with self.graph.as_default():
        #self.X = tf.placeholder("float", [None,x_t.shape[1],1])
        #self.build_model()
        #init = tf.global_variables_initializer()
        #self.sess = tf.Session()
        #self.sess.run(init)

        return self.sess.run(self.predictions, feed_dict={self.X: x_t})
        
    def get_latent(self, x_t):
        # Returns the latent representation of the input signal
        return self.sess.run(self.layer_2, feed_dict={self.X: x_t})

    def get_unpool(self, x_t):
        return self.sess.run(self.unpool_1, feed_dict={self.X: x_t})

    def interpolate(self, s):
        #self.layer_3 = tf.transpose(self.layer_3, perm=(0,2,1))
        # return tf.stack([tf.gather(self.layer_3, s, axis=1),
        # (tf.gather(self.layer_3, s, axis=1) + tf.gather(self.layer_3, s-1, axis=1)) / 2], axis=0)#, axis=1)#, axis=1)#, (-1,1,tf.shape(self.layer_3)[2]*2))
        #return (self.layer_3[:,s,:] + self.layer_3[:,s - 1,:]) / 2
        # return tf.stack([self.layer_3[:,s,:], (self.layer_3[:,s,:] + self.layer_3[:,s - 1,:]) / 2], axis=1)#,(-1, 1, tf.shape(self.layer_3)[2] * 2))
        # return tf.stack([self.layer_3[:,s,:], (self.layer_3[:,s,:] + self.layer_3[:,s - 1,:]) / 2], axis=0)#,(-1, 1, tf.shape(self.layer_3)[2] * 2))
        # index = s*2
        # self.unpool_1[:, index, :].assign(self.layer_3[:, s, :])
        # self.unpool_1[:, index - 1, :].assign((self.layer_3[:, s, :] + self.layer_3[:, s - 1, :]) / 2)
        # return 1
        # return tf.concat([(self.layer_3[:, s, :] + self.layer_3[:, s - 1, :]) / 2, self.layer_3[:, s, :]], axis=1)
        return tf.transpose(tf.stack([(self.layer_3[:, s, :] + self.layer_3[:, s - 1, :]) / 2, self.layer_3[:, s, :]], axis=1), perm=[1, 0, 2])
        # return index