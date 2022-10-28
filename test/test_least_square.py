import odak
import sys
import numpy as np


def function(x, parameters):
    y = parameters[0] * x + parameters[1]
    return y


def test():
    x  = np.linspace(0, 100, 20)
    parameters = [4, 1]
    y  = function(x, parameters)
    estimated_parameters = odak.fit.least_square_1d(x, y)
    estimations = function(x, estimated_parameters)
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
