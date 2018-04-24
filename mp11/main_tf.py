"""Generative Adversarial Networks
"""

import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.gan import Gan

def train(model, mnist_dataset, learning_rate=0.0001, batch_size=32,
          num_steps=800, nlatent=2, epoch=0):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size and
    learning_rate.

    Args:
        model(GAN): Initialized generative network.
        mnist_dataset: input_data.
        learning_rate(float): Learning rate.
        batch_size(int): batch size used for training.
        num_steps(int): Number of steps to run the update ops.
    """
    d_iters = 1
    g_iters = 8
    for step in range(num_steps):
        for _ in range(d_iters):
            batch_x, _ = mnist_dataset.train.next_batch(batch_size)
            model.session.run(
                model.update_d_op,
                feed_dict={
                    model.x_placeholder: batch_x,
                    model.z_placeholder: np.random.uniform(-1,1,[batch_size,nlatent]),
                    model.learning_rate_placeholder: learning_rate
                }
            )

        for _ in range(g_iters):
            batch_x, _ = mnist_dataset.train.next_batch(batch_size)
            model.session.run(
                model.update_g_op,
                feed_dict={
                    model.x_placeholder: batch_x,
                    model.z_placeholder: np.random.uniform(-1,1,[batch_size,nlatent]),
                    model.learning_rate_placeholder: learning_rate
                }
            )

        if step % 100 == 0:
            summary = model.summary_op
            batch_x, _ = mnist_dataset.train.next_batch(batch_size)
            summary = model.session.run(
                model.summary_op,
                feed_dict={
                    model.x_placeholder: batch_x,
                    model.z_placeholder: np.random.uniform(-1,1,[batch_size,nlatent]),
                    model.learning_rate_placeholder: learning_rate
                }
            )
            model.writer.add_summary(summary, step + epoch * num_steps)

def generate_image(model, nlatent=2):
    n_images = 5
    x_z = np.linspace(-1, 1, n_images)
    y_z = np.linspace(-1, 1, n_images)

    out = np.empty((28*n_images, 28*n_images))
    for x_idx, x in enumerate(x_z):
        for y_idx, y in enumerate(y_z):
            z = []
            if nlatent == 2:
                z = np.array([[y, x]], dtype=np.float32)
            else:
                z = np.array(np.random.uniform(-1,1,[1,nlatent]), dtype=np.float32)
            z = tf.convert_to_tensor(z, np.float32)
            img = model._generator(z, reuse=True).eval()
            out[x_idx*28:(x_idx+1)*28,
                y_idx*28:(y_idx+1)*28] = img.reshape((28, 28))
    plt.imsave('latent_space_gan.png', out, cmap="gray")


def main(_):
    """High level pipeline.

    This scripts performs the training for GANs.
    """
    # Get dataset.
    mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    nlatent = 2

    # Build model.
    model = Gan(nlatent=nlatent)

    # Start training.
    for i in range(10):
        print("epoch", i)
        train(model, mnist_dataset, nlatent=nlatent, epoch=i)

        # Plot out latent space, for +/- 1.
        print("generating image", i)
        generate_image(model, nlatent=nlatent)

if __name__ == "__main__":
    tf.app.run()
