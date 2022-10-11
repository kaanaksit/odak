import numpy as np
from tqdm import tqdm

def least_square_1d(x, y):
    """
    A function to fit a line to given x and y data (y=mx+n). Inspired from: https://mmas.github.io/least-squares-fitting-numpy-scipy

    Parameters
    ----------
    x          : numpy.array
                 1D input data.
    y          : numpy.array
                 1D output data.

    Returns
    -------
    parameters : numpy.array
                 Parameters of m and n in a line (y=mx+n).
    """
    w = np.vstack([x, np.ones(x.shape[0])]).T
    parameters = np.dot(np.linalg.inv(np.dot(w.T, w)), np.dot(w.T, y))
    return parameters


def gradient_descent_1d(
                     input_data,
                     ground_truth_data,
                     parameters,
                     function,
                     gradient_function,
                     loss_function,
                     learning_rate=1e-1,
                     iteration_number=10
                    ):
    """
    Vanilla Gradient Descent algorithm for 1D data.
    """
    t = tqdm(range(iteration_number))
    for i in t:
        gradient = np.zeros(parameters.shape[0])
        for j in range(input_data.shape[0]):
            x = input_data[j]
            y = ground_truth_data[j]
            gradient = gradient + gradient_function(x, y, function, parameters)
        parameters = parameters - learning_rate * gradient / input_data.shape[0]
        loss = loss_function(ground_truth_data, function(input_data, parameters))
        description = 'Iteration number:{}, loss:{:0.4f}, parameters:{}'.format(i, loss, np.round(parameters, 2))
        t.set_description(description)
    return parameters
