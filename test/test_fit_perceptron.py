import numpy as np
import sys
import odak

def generate_data(number_of_elements, offset=[0, 0], scale=[1., 1.], label=0.):
    """
    A function to generate reasonable data with a specific class.

    Parameter
    ---------
    number_of_elements  : int
                          Number of items in the final data [m].
    offset              : list
                          Offset along X and Y axes in the generated data.
    scale               : list
                          Scaling factor for points generated along X and Y axes.
    label               : float
                          Choose your class either zero or one.

    Return
    ------
    data                : numpy.array
                          Values that represents the generated data [m x 2].
    labels              : numpy.array
                          Labels for the generated data [m x 1].
    """
    points_x = np.random.rand(number_of_elements, 1) * scale[0] + offset[0]
    points_y = np.random.rand(number_of_elements, 1) * scale[1] + offset[1]
    data = np.hstack((points_x, points_y))
    labels = np.ones((number_of_elements, 1)) * label
    return data, labels


def plot_data(x, y, w, show=True):
    """
    Function to plot data and a boundary drawn using the given weights.
    """
    if show == False:
        return
    import matplotlib.pyplot as plt
    figure = plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=y[:, 0])
    x1 = [min(x[:, 0]), max(x[:, 0])]
    m = - w[1] / w[2]
    n = - w[0] / w[2]
    x2 = m * x1 + n
    plt.plot(x1, x2)
    plt.show()


def test():
    x0, y0 = generate_data(100, offset=[ 4.,  4.], scale=[4., 4.], label=0)
    x1, y1 = generate_data(100, offset=[-4., -4.], scale=[5., 5.], label=1) 
    x_train = np.vstack((x0, x1)); y_train = np.vstack((y0, y1))
    learning_rate = 0.1; iteration_number = 100
    weights = odak.fit.perceptron(
                                  x_train,
                                  y_train,
                                  learning_rate=learning_rate,
                                  iteration_number=iteration_number
                                 ) 
    plot_data(x_train, y_train, weights, show=False)
    assert True == True


if __name__ == '__main__':
    sys.exit(test())

