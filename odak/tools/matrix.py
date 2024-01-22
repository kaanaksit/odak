import numpy as np


def create_empty_list(dimensions = [1, 1]):
    """
    A definition to create an empty Pythonic list.

    Parameters
    ----------
    dimensions   : list
                   Dimensions of the list to be created.

    Returns
    -------
    new_list     : list
                   New empty list.
    """
    new_list = 0
    for n in reversed(dimensions):
        new_list = [new_list] * n
    return new_list


def nufft2(field, fx, fy, size=None, sign=1, eps=10**(-12)):
    """
    A definition to take 2D Non-Uniform Fast Fourier Transform (NUFFT).

    Parameters
    ----------
    field       : ndarray
                  Input field.
    fx          : ndarray
                  Frequencies along x axis.
    fy          : ndarray
                  Frequencies along y axis.
    size        : list
                  Size.
    sign        : float
                  Sign of the exponential used in NUFFT kernel.
    eps         : float
                  Accuracy of NUFFT.

    Returns
    ----------
    result      : ndarray
                  Inverse NUFFT of the input field.
    """
    try:
        import finufft
    except:
        print('odak.tools.nufft2 requires finufft to be installed: pip install finufft')
    image = np.copy(field).astype(np.complex128)
    result = finufft.nufft2d2(
        fx.flatten(), fy.flatten(), image, eps=eps, isign=sign)
    if type(size) == type(None):
        result = result.reshape(field.shape)
    else:
        result = result.reshape(size)
    return result


def nuifft2(field, fx, fy, size=None, sign=1, eps=10**(-12)):
    """
    A definition to take 2D Adjoint Non-Uniform Fast Fourier Transform (NUFFT).

    Parameters
    ----------
    field       : ndarray
                  Input field.
    fx          : ndarray
                  Frequencies along x axis.
    fy          : ndarray
                  Frequencies along y axis.
    size        : list or ndarray
                  Shape of the NUFFT calculated for an input field.
    sign        : float
                  Sign of the exponential used in NUFFT kernel.
    eps         : float
                  Accuracy of NUFFT.

    Returns
    ----------
    result      : ndarray
                  NUFFT of the input field.
    """
    try:
        import finufft
    except:
        print('odak.tools.nuifft2 requires finufft to be installed: pip install finufft')
    image = np.copy(field).astype(np.complex128)
    if type(size) == type(None):
        result = finufft.nufft2d1(
            fx.flatten(),
            fy.flatten(),
            image.flatten(),
            image.shape,
            eps=eps,
            isign=sign
        )
    else:
        result = finufft.nufft2d1(
            fx.flatten(),
            fy.flatten(),
            image.flatten(),
            (size[0], size[1]),
            eps=eps,
            isign=sign
        )
    result = np.asarray(result)
    return result


def generate_bandlimits(size=[512, 512], levels=9):
    """
    A definition to calculate octaves used in bandlimiting frequencies in the frequency domain.

    Parameters
    ----------
    size       : list
                 Size of each mask in octaves.

    Returns
    ----------
    masks      : ndarray
                 Masks (Octaves).
    """
    masks = np.zeros((levels, size[0], size[1]))
    cx = int(size[0]/2)
    cy = int(size[1]/2)
    for i in range(0, masks.shape[0]):
        deltax = int((size[0])/(2**(i+1)))
        deltay = int((size[1])/(2**(i+1)))
        masks[
            i,
            cx-deltax:cx+deltax,
            cy-deltay:cy+deltay
        ] = 1.
        masks[
            i,
            int(cx-deltax/2.):int(cx+deltax/2.),
            int(cy-deltay/2.):int(cy+deltay/2.)
        ] = 0.
    masks = np.asarray(masks)
    return masks


def zero_pad(field, size=None, method='center'):
    """
    Definition to zero pad a MxN array to 2Mx2N array.

    Parameters
    ----------
    field             : ndarray
                        Input field MxN array.
    size              : list
                        Size to be zeropadded.
    method            : str
                        Zeropad either by placing the content to center or to the left.

    Returns
    ----------
    field_zero_padded : ndarray
                        Zeropadded version of the input field.
    """
    if type(size) == type(None):
        hx = int(np.ceil(field.shape[0])/2)
        hy = int(np.ceil(field.shape[1])/2)
    else:
        hx = int(np.ceil((size[0]-field.shape[0])/2))
        hy = int(np.ceil((size[1]-field.shape[1])/2))
    if method == 'center':
        field_zero_padded = np.pad(
            field, ([hx, hx], [hy, hy]), constant_values=(0, 0))
    elif method == 'left aligned':
        field_zero_padded = np.pad(
            field, ([0, 2*hx], [0, 2*hy]), constant_values=(0, 0))
    if type(size) != type(None):
        field_zero_padded = field_zero_padded[0:size[0], 0:size[1]]
    return field_zero_padded


def crop_center(field, size=None):
    """
    Definition to crop the center of a field with 2Mx2N size. The outcome is a MxN array.

    Parameters
    ----------
    field       : ndarray
                  Input field 2Mx2N array.

    Returns
    ----------
    cropped     : ndarray
                  Cropped version of the input field.
    """
    if type(size) == type(None):
        qx = int(np.ceil(field.shape[0])/4)
        qy = int(np.ceil(field.shape[1])/4)
        cropped = np.copy(field[qx:3*qx, qy:3*qy])
    else:
        cx = int(np.ceil(field.shape[0]/2))
        cy = int(np.ceil(field.shape[1]/2))
        hx = int(np.ceil(size[0]/2))
        hy = int(np.ceil(size[1]/2))
        cropped = np.copy(field[cx-hx:cx+hx, cy-hy:cy+hy])
    return cropped


def quantize(image_field, bits=4):
    """
    Definitio to quantize a image field (0-255, 8 bit) to a certain bits level.

    Parameters
    ----------
    image_field : ndarray
                  Input image field.
    bits        : int
                  A value in between 0 to 8. Can not be zero.

    Returns
    ----------
    new_field   : ndarray
                  Quantized image field.
    """
    divider = 2**(8-bits)
    new_field = image_field/divider
    new_field = new_field.astype(np.int64)
    return new_field


def convolve2d(field, kernel):
    """
    Definition to convolve a field with a kernel by multiplying in frequency space.

    Parameters
    ----------
    field       : ndarray
                  Input field with MxN shape.
    kernel      : ndarray
                  Input kernel with MxN shape.

    Returns
    ----------
    new_field   : ndarray
                  Convolved field.
    """
    fr = np.fft.fft2(field)
    fr2 = np.fft.fft2(np.flipud(np.fliplr(kernel)))
    m, n = fr.shape
    new_field = np.real(np.fft.ifft2(fr*fr2))
    new_field = np.roll(new_field, int(-m/2+1), axis=0)
    new_field = np.roll(new_field, int(-n/2+1), axis=1)
    return new_field


def generate_2d_gaussian(kernel_length=[21, 21], nsigma=[3, 3]):
    """
    Generate 2D Gaussian kernel. Inspired from https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy

    Parameters
    ----------
    kernel_length : list
                    Length of the Gaussian kernel along X and Y axes.
    nsigma        : list
                    Sigma of the Gaussian kernel along X and Y axes.

    Returns
    ----------
    kernel_2d     : ndarray
                    Generated Gaussian kernel.
    """
    x = np.linspace(-nsigma[0], nsigma[0], kernel_length[0]+1)
    y = np.linspace(-nsigma[1], nsigma[1], kernel_length[1]+1)
    xx, yy = np.meshgrid(x, y)
    kernel_2d = np.exp(-0.5*(np.square(xx) /
                       np.square(nsigma[0]) + np.square(yy)/np.square(nsigma[1])))
    kernel_2d = kernel_2d/kernel_2d.sum()
    return kernel_2d


def blur_gaussian(field, kernel_length=[21, 21], nsigma=[3, 3]):
    """
    A definition to blur a field using a Gaussian kernel.

    Parameters
    ----------
    field         : ndarray
                    MxN field.
    kernel_length : list
                    Length of the Gaussian kernel along X and Y axes.
    nsigma        : list
                    Sigma of the Gaussian kernel along X and Y axes.

    Returns
    ----------
    blurred_field : ndarray
                    Blurred field.
    """
    kernel = generate_2d_gaussian(kernel_length, nsigma)
    kernel = zero_pad(kernel, field.shape)
    blurred_field = convolve2d(field, kernel)
    blurred_field = blurred_field/np.amax(blurred_field)
    return blurred_field
