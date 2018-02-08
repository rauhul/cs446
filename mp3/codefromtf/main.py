"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.001
max_iters = 10000

def main(_):
    # Load dataset.
    X, Y = read_dataset_tf('../data/trainset','indexing.txt')

    # Initialize model.
    model = LogisticModel_TF(16, 'gaussian')

    # Build TensorFlow training graph
    model.build_graph(learn_rate)

    # Train model via gradient descent.
    Y_guess = model.fit(Y, X, max_iters)

    # Compute classification accuracy based on the return of the "fit" method
    correct = 0
    for i in range(len(Y)):
        elem_guess = (Y_guess[0][i][0] > 0.5) * 2 - 1
        if Y[i][0] == elem_guess:
            correct += 1

    accuracy = correct / len(Y)

    print(accuracy)

if __name__ == '__main__':
    tf.app.run()
