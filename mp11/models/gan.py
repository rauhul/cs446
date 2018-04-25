"""Generative adversarial network."""

import numpy as np
import tensorflow as tf

class Gan(object):
    """Adversary based generator network.
    """
    def __init__(self, ndims=784, nlatent=2):
        """Initializes a GAN

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """
        self._ndims   = ndims
        self._nlatent = nlatent

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])

        # Input noise
        self.z_placeholder = tf.placeholder(tf.float32, [None, nlatent])

        # Input learning rate
        self.learning_rate_placeholder = tf.placeholder(tf.float32)

        # Build graph.
        self.x_hat = self._generator(self.z_placeholder)
        y_hat      = self._discriminator(self.x_hat)
        y          = self._discriminator(self.x_placeholder, reuse=True)

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)

        # Add optimizers for appropriate variables
        self.update_d_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder).minimize(
            self.d_loss,
            var_list=[var for var in tf.global_variables() if "discriminator" in var.name]
        )
        self.update_g_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder).minimize(
            self.g_loss,
            var_list=[var for var in tf.global_variables() if "generator" in var.name]
        )

        # Tensorboard
        self.writer = tf.summary.FileWriter("/Users/rauhul/Desktop/tensorboard_logs/", graph=tf.get_default_graph())
        self.summary_op = tf.summary.merge_all()

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake image(None, 1).
              DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.
        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            d_full_1 = tf.nn.dropout(tf.layers.dense(x, 64, activation=tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer(stddev=0.02)), 0.2)
            d_full_2 = tf.layers.dense(d_full_1, 1, activation=None)
            return d_full_2

    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator.

        Args:
            y (tf.Tensor): The output tensor of the discriminator for true images of dimension (None, 1).
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.
        """
        real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y,     labels=tf.ones_like(y)))
        fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=tf.zeros_like(y_hat)))

        tf.summary.scalar('discriminator_cross_entropy_real', real)
        tf.summary.scalar('discriminator_cross_entropy_fake', fake)

        return real + fake

    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:
            g_full_1 = tf.layers.dense(z, 14*14*16, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            g_full_1_norm = tf.layers.batch_normalization(g_full_1)
            g_full_1_acti = tf.nn.leaky_relu(g_full_1_norm)


            square = tf.reshape(g_full_1_acti, [-1, 14, 14, 16])
            g_conv_1 = tf.layers.conv2d_transpose(
                inputs=square,
                filters=64,
                strides=(2, 2),
                kernel_size=[5, 5],
                padding="same",
            )
            g_conv_1_norm = tf.layers.batch_normalization(g_conv_1)
            g_conv_1_acti = tf.nn.leaky_relu(g_conv_1_norm)


            g_conv_2 = tf.layers.conv2d_transpose(
                inputs=g_conv_1_acti,
                filters=64,
                strides=(1, 1),
                kernel_size=[5, 5],
                padding="same",
            )
            g_conv_2_norm = tf.layers.batch_normalization(g_conv_2)
            g_conv_2_acti = tf.nn.leaky_relu(g_conv_2_norm)


            g_conv_3 = tf.layers.conv2d_transpose(
                inputs=g_conv_2_acti,
                filters=1,
                strides=(1, 1),
                kernel_size=[1, 1],
                padding="same",
            )
            g_conv_3_acti = tf.nn.sigmoid(g_conv_3)
            g_conv_3_flat = tf.layers.flatten(g_conv_3_acti)
            return g_conv_3_flat

    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.
        """

        # move towards 1
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=tf.ones_like(y_hat)))
        tf.summary.scalar('generator_cross_entropy', g_loss)
        return g_loss
