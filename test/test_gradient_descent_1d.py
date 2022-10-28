import numpy as np
import sys
import odak


def gradient_function(x, y, function, parameters):
    solution = function(x, parameters)
    gradient = np.array([
                         -2 * x**2 * (y - solution),
                         -2 * x * (y- solution),
                         -2 * (y - solution)
                        ])
    return gradient


def function(x, parameters):
    y = parameters[0] * x**2 + parameters[1] * x + parameters[2]
    return y


def l2_loss(a, b):
    loss = np.sum((a - b)**2)
    return loss


def test():
    x = np.linspace(0, 1., 20) 
    y = function(x, parameters=[2., 1., 10.])

    learning_rate = 5e-1
    iteration_number = 2000
    initial_parameters = np.array([10., 10., 0.])
    estimated_parameters = odak.fit.gradient_descent_1d(
                                                        input_data=x,
                                                        ground_truth_data=y,
                                                        function=function,
                                                        loss_function=l2_loss,
                                                        gradient_function=gradient_function,
                                                        parameters=initial_parameters,
                                                        learning_rate=learning_rate,
                                                        iteration_number=iteration_number
                                                       )
    assert True == True


if __name__ == '__main__':
   sys.exit(test())
