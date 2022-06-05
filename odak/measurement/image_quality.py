import numpy as np


def line_spread_function(line):
    """
    Definition to take the gradient of a 1D function.

    Parameters
    ----------
    line          : ndarray
                    1D array.

    Returns
    ----------
    result        : ndarray
                    Gradient of the given 1D array.
    """
    result = np.gradient(line)
    result = np.asarray(result)
    return result


def fourier_transform_1d(line):
    """
    Definition to take the 1D fourier transform.
    This is used only for modulation transfer function calculations.

    Parameters
    ----------
    line         : ndarray
                   1D array.

    Returns
    ----------
    result       : ndarray
                   Positive side of the fourier transform of the given line.
    """
    result = np.fft.fft(line)
    result /= np.amax(result)
    result = result[np.arange(0, int(line.shape[0]/2))]
    return result


def polynomial_fit(line_x, line_y, fit_degree=3):
    """
    Definition to fit polynomials to a vector.

    Parameters
    ----------
    line_x      : ndarray
                  Values along X axis.
    line_y      : ndarray
                  Values along Y axis.
    degree      : int
                  Degree of the polynomial fit.

    Returns
    ----------
    p           : numpy.poly1d
                  polynomial fit.
    """
    if np.__name__ == 'numpy':
        fun_poly = np.polyfit(line_x, line_y, fit_degree)
        p = np.poly1d(fun_poly)
    else:
        import numpy
        line_x = np.asnumpy(line_x)
        line_y = np.asnumpy(line_y)
        fun_poly = numpy.polyfit(line_x, line_y, fit_degree)
        p = numpy.poly1d(fun_poly)
    return p


def roi(image, location=[0, 100, 0, 100], threshold=[0, 1, 0, 1]):
    """
    Definition to get the lines from a target ROI.

    Parameters
    ----------
    image      : ndarray
                 a 2D image to be sliced (nxm).
    location   : ndarray
                 Locations for taking the ROI.
    threshold  : list
                 Threshold below and above these numbers.

    Returns
    -------
    line_x     : ndarray
                 Line slice.
    line_y     : ndarray
                 Line slice.
    """
    img = image[location[0]:location[1], location[2]:location[3]]
    if len(img.shape) == 3:
        img = np.sum(img, axis=2)
    line_x = img[:, int(img.shape[1]/2)]
    line_y = img[int(img.shape[0]/2), :]
    line_x = np.asarray(line_x)
    line_y = np.asarray(line_y)
    line_x = line_x - np.amin(line_x)
    line_x = line_x / np.amax(line_x)
    line_y = line_y - np.amin(line_y)
    line_y = line_y / np.amax(line_y)
    line_x[line_x < threshold[0]] = 0
    line_x[line_x > threshold[1]] = 1
    line_y[line_y < threshold[2]] = 0
    line_y[line_y > threshold[3]] = 1
    return line_x, line_y, img


def modulation_transfer_function(line_x, line_y, px_size):
    """
    Definition to compute modulation transfer function.
    This definition is based on the work by Peter Burns.
    For more consult Burns, Peter D. "Slanted-edge MTF for digital camera
    and scanner analysis." Is and Ts Pics Conference.
    SOCIETY FOR IMAGING SCIENCE & TECHNOLOGY, 2000.

    Parameters
    ----------
    line_x       : ndarray
                   Slice of an image along X axis. See odak.measurements.roi().
    line_y       : ndarray
                   Slice of an image along Y axis. See odak.measurements.roi().
    px_size      : ndarray
                   Physical angular size of each pixels on the image plane.

    Returns
    ----------
    mtf          : ndarray
                   Calculated modulation transfer function along X and Y axes.
    frq          : ndarray
                   Frequencies of the calculated MTF.
    """
    der_x = line_spread_function(line_x)
    der_y = line_spread_function(line_y)
    mtf_x = fourier_transform_1d(der_x)
    mtf_y = fourier_transform_1d(der_y)
    n_x = len(der_x)
    k_x = np.arange(n_x)
    T_x = n_x*px_size[0]
    frq_x = k_x/T_x
    frq_x = frq_x[np.arange(0, int(n_x/2))]
    n_y = len(der_y)
    k_y = np.arange(n_y)
    T_y = n_y*px_size[1]
    frq_y = k_y/T_y
    frq_y = frq_y[np.arange(0, int(n_y/2))]
    return [mtf_x, mtf_y], [frq_x, frq_y]
