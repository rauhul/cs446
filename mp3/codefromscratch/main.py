"""Main function for binary classifier"""

import numpy as np

from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.001
max_iters = 100

if __name__ == '__main__':

    # Load dataset.
    X, Y = read_dataset('../data/trainset','indexing.txt')

    # Initialize model.
    model = LogisticModel(16, 'gaussian')

    # Train model via gradient descent.
    model.fit(Y, X, learn_rate, max_iters)

    # Save trained model to 'trained_weights.np'
    model.save_model('trained_weights.np')

    # Load trained model from 'trained_weights.np'
    model.load_model('trained_weights.np')

    # Try all other methods: forward, backward, classify, compute accuracy

    # Compute classification accuracy based on the return of the "fit" method
    Y_guess = model.classify(X)
    equal = np.equal(Y, Y_guess)
    accuracy = np.mean(equal)

    print(accuracy)


