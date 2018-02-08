"""logistic model class for binary classification."""
import tensorflow as tf
import numpy as np

class LogisticModel_TF(object):

    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of Weight is the bias term,
            Weight = [Bias, W1, W2, W3, ...]
            where Wi correspnds to each feature dimension

        W_init needs to support:
          'zeros': initialize self.W with all zeros.
          'ones': initialze self.W with all ones.
          'uniform': initialize self.W with uniform random number between [0,1)
          'gaussian': initialize self.W with gaussion distribution (0, 0.1)

        Args:
            ndims(int): feature dimension
            W_init(str): types of initialization.
        """
        self.ndims = ndims
        self.W_init = W_init
        self.W0 = None

        if W_init == 'zeros':
            self.W0 = tf.zeros([self.ndims+1,1])
        elif W_init == 'ones':
            self.W0 = tf.ones([self.ndims+1,1])
        elif W_init == 'uniform':
            self.W0 = tf.random_uniform([self.ndims+1,1])
        elif W_init == 'gaussian':
            self.W0 = tf.random_normal([self.ndims+1,1], stddev=0.1)
        else:
            print ('Unknown W_init ', W_init)


    def build_graph(self, learn_rate):
        """ build tensorflow training graph for logistic model.
        Args:
            learn_rate: learn rate for gradient descent
            ......: append as many arguments as you want
        """

        self.W = tf.Variable(self.W0)
        self.X = tf.placeholder(tf.float32, [None, self.ndims+1])
        self.Y = tf.placeholder(tf.float32, [None, 1])

        self.Y_ = tf.tensordot(self.X, self.W, 1)
        self.Y__ = tf.sigmoid(self.Y_)

        self.L = tf.subtract(self.Y, self.Y__)
        self.L_ = tf.square(self.L)
        self.L__ = tf.reduce_sum(self.L_)

        self.O = tf.train.AdamOptimizer(learn_rate).minimize(self.L__)

    def fit(self, Y_true, X, max_iters):
        """ train model with input dataset using gradient descent.
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,1)
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            max_iters: maximal number of training iterations
            ......: append as many arguments as you want
        Returns:
            (numpy.ndarray): sigmoid output from well trained logistic model, used for classification
                             with a dimension of (# of samples, 1)
        """
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(max_iters):
                _, L = sess.run([self.O, self.L__], feed_dict={self.X:X, self.Y:Y_true})

            return sess.run([self.Y__], feed_dict={self.X:X, self.Y:Y_true})
