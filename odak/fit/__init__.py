"""
``odak.fit``
===================
Provides functions to fit models to a provided data. These functions could be best described as a catalog of machine learning models.
"""

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
    
    Parameters
    ----------
    input_data        : torch.tensor
                        One-dimensional input data.
    ground_truth_data : torch.tensor
                        One-dimensional ground truth data.
    parameters        : torch.tensor
                        Parameters to be optimized.
    function          : function
                        Function to estimate an output using the parameters.
    gradient_function : function
                        Function used in estimating gradient to update parameters at each iteration.
    learning rate     : float
                        Learning rate.
    iteration_number  : int
                        Iteration number.
                        
    Returns
    -------
    parameters        : torch.tensor
                        Optimized/estimated parameters.
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


def perceptron(x, y, learning_rate=0.1, iteration_number=100):
    """
    A function to train a perceptron model.

    Parameters
    ----------
    x                : numpy.array
                       Input X-Y pairs [m x 2].
    y                : numpy.array
                       Labels for the input data [m x 1]
    learning_rate    : float
                       Learning rate.
    iteration_number : int
                       Iteration number.

    Returns
    -------
    weights          : numpy.array
                       Trained weights of our model [3 x 1].
    """
    weights = np.zeros((x.shape[1] + 1, 1))
    t = tqdm(range(iteration_number))
    for step in t:
        unsuccessful = 0
        for data_id in range(x.shape[0]):
            x_i = np.insert(x[data_id], 0, 1).reshape(-1, 1)
            y_i = y[data_id]
            y_hat = threshold_linear_model(x_i, weights)
            if y_hat - y_i != 0:
                unsuccessful += 1
                weights = weights + learning_rate * (y_i - y_hat) * x_i 
            description = 'Unsuccessful count: {}/{}'.format(unsuccessful, x.shape[0])
    return weights


def threshold_linear_model(x, w, threshold=0):
    """
    A function for thresholding a linear model described with a dot product.

    Parameters
    ----------
    x                : numpy.array
                       Input data [3 x 1].
    w                : numpy.array
                       Weights [3 x 1].
    threshold        : float
                       Value for thresholding.

    Returns
    -------
    result           : int
                       Estimated class of the input data. It could either be one or zero.
    """
    value = np.dot(x.T, w)
    result = 0
    if value >= threshold:
       result = 1
    return result
