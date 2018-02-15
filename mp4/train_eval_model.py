"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers

def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    images = data['image']
    labels = data['label']

    for step in range(num_steps):
        start_idx = step*batch_size
        end_idx = start_idx+batch_size
        if start_idx >= len(images):
            break
        if end_idx >= len(images):
            end_idx = len(images)
        indices = np.arange(start_idx, end_idx)
        if shuffle:
            np.random.shuffle(indices)
        update_step(images[indices], labels[indices], model, learning_rate)

    return model

def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    model.w -= learning_rate * model.backward(model.forward(x_batch), y_batch)

def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])

    W, _ = model.w.shape
    model.w = z[0:W]

def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    images = data['image']
    labels = data['label']
    N, F = images.shape

    P = np.identity(N + F + 1)
    P[F,F] = 0

    q = np.zeros((N + F + 1, 1))

    ones = np.ones((N, 1))
    images_ext = np.concatenate((images, ones), axis=1)
    G = images_ext * -labels
    G = np.concatenate((G, np.identity(N)), axis=1)

    h = -np.ones((N, 1))
    return P, q, G, h

def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    images = data['image']
    labels = data['label']

    forward_result = model.forward(images)
    predictions = model.predict(forward_result)
    correct_predictions = predictions == labels

    loss = model.total_loss(forward_result, labels)
    acc = np.mean(correct_predictions)
    return loss, acc
