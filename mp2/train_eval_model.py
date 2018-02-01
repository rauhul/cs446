"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression


def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    for step in range(num_steps):
        start_idx = step*batch_size
        end_idx = start_idx+batch_size
        if start_idx >= len(processed_dataset[0]):
            break
        if end_idx >= len(processed_dataset[0]):
            end_idx = len(processed_dataset[0])
        indices = np.arange(start_idx, end_idx)
        if shuffle:
            np.random.shuffle(indices)
        update_step(processed_dataset[0][indices], processed_dataset[1][indices], model, learning_rate)

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    model.w -= learning_rate * model.backward(model.forward(x_batch), y_batch)

def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    d = processed_dataset
    one_postpend = np.hstack((d[0], np.ones((len(d[0]),1))))
    model.w, _, _, _ = np.linalg.lstsq(one_postpend, d[1], rcond=None)

def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    return model.total_loss(model.forward(processed_dataset[0]), processed_dataset[1])
