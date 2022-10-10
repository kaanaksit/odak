import numpy as np


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

