import numpy as np
import torch
import torch.nn


def quantize(image_field, bits=4):
    """ 
    Definition to quantize a image field (0-255, 8 bit) to a certain bits level.

    Parameters
    ----------
    image_field : torch.tensor
                  Input image field.
    bits        : int
                  A value in between 0 to 8. Can not be zero.

    Returns
    ----------
    new_field   : torch.tensor
                  Quantized image field.
    """
    divider = 2**(8-bits)
    new_field = image_field/divider
    new_field = new_field.int()
    return new_field


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
        hx = int(torch.ceil(torch.tensor([field.shape[0]/2])))
        hy = int(torch.ceil(torch.tensor([field.shape[1]/2])))
    else:
        hx = int(torch.ceil(torch.tensor([(size[0]-field.shape[0])/2])))
        hy = int(torch.ceil(torch.tensor([(size[1]-field.shape[1])/2])))
    if method == 'center':
        m = torch.nn.ZeroPad2d((hy, hy, hx, hx))
    elif method == 'left aligned':
        m = torch.nn.ZeroPad2d((0, hy*2, 0, hx*2))
    field_zero_padded = m(field)
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
    size        : list
                  Dimensions to crop with respect to center of the image.

    Returns
    ----------
    cropped     : ndarray
                  Cropped version of the input field.
    """
    if type(size) == type(None):
        qx = int(torch.ceil(torch.tensor(field.shape[0])/4))
        qy = int(torch.ceil(torch.tensor(field.shape[1])/4))
        cropped = field[qx:3*qx, qy:3*qy]
    else:
        cx = int(torch.ceil(torch.tensor(field.shape[0]/2)))
        cy = int(torch.ceil(torch.tensor(field.shape[1]/2)))
        hx = int(torch.ceil(torch.tensor(size[0]/2)))
        hy = int(torch.ceil(torch.tensor(size[1]/2)))
        cropped = field[cx-hx:cx+hx, cy-hy:cy+hy]
    return cropped


def convolve2d(field, kernel):
    """
    Definition to convolve a field with a kernel by multiplying in frequency space.

    Parameters
    ----------
    field       : torch.tensor
                  Input field with MxN shape.
    kernel      : torch.tensor
                  Input kernel with MxN shape.

    Returns
    ----------
    new_field   : torch.tensor
                  Convolved field.
    """
    fr = torch.fft.fft2(field)
    fr2 = torch.fft.fft2(torch.flip(torch.flip(kernel, [1, 0]), [0, 1]))
    m, n = fr.shape
    new_field = torch.real(torch.fft.ifft2(fr*fr2))
    new_field = torch.roll(new_field, shifts=(int(n/2+1), 0), dims=(1, 0))
    new_field = torch.roll(new_field, shifts=(int(m/2+1), 0), dims=(0, 1))
    return new_field


def generate_2d_gaussian(kernel_length=[21, 21], nsigma=[3, 3], mu=[0, 0], normalize=False):
    """
    Generate 2D Gaussian kernel. Inspired from https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy

    Parameters
    ----------
    kernel_length : list
                    Length of the Gaussian kernel along X and Y axes.
    nsigma        : list
                    Sigma of the Gaussian kernel along X and Y axes.
    mu            : list
                    Mu of the Gaussian kernel along X and Y axes.
    normalize     : bool
                    If set True, normalize the output.

    Returns
    ----------
    kernel_2d     : torch.tensor
                    Generated Gaussian kernel.
    """
    x = torch.linspace(-kernel_length[0]/2., kernel_length[0]/2., kernel_length[0])
    y = torch.linspace(-kernel_length[1]/2., kernel_length[1]/2., kernel_length[1])
    X, Y = torch.meshgrid(x, y, indexing='ij')
    if nsigma[0] == 0:
        nsigma[0] = 1e-5
    if nsigma[1] == 0:
        nsigma[1] = 1e-5
    kernel_2d = 1. / (2. * np.pi * nsigma[0] * nsigma[1]) * torch.exp(-((X - mu[0])**2. / (2. * nsigma[0]**2.) + (Y - mu[1])**2. / (2. * nsigma[1]**2.)))
    if normalize:
        kernel_2d = kernel_2d / kernel_2d.max()
    return kernel_2d


def blur_gaussian(field, kernel_length=[21, 21], nsigma=[3, 3], padding='same'):
    """
    A definition to blur a field using a Gaussian kernel.

    Parameters
    ----------
    field         : torch.tensor
                    MxN field.
    kernel_length : list
                    Length of the Gaussian kernel along X and Y axes.
    nsigma        : list
                    Sigma of the Gaussian kernel along X and Y axes.
    padding       : int or string
                    Padding value, see torch.nn.functional.conv2d() for more.

    Returns
    ----------
    blurred_field : torch.tensor
                    Blurred field.
    """
    kernel = generate_2d_gaussian(kernel_length, nsigma).to(field.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    if len(field.shape) == 2:
        field = field.view(1, 1, field.shape[-2], field.shape[-1])
    blurred_field = torch.nn.functional.conv2d(field, kernel, padding='same')
    if field.shape[1] == 1:
        blurred_field = blurred_field.view(
                                           blurred_field.shape[-2],
                                           blurred_field.shape[-1]
                                          )
    return blurred_field


def convert_rgb_to_yuv(image):
    """
    Definition to convert red, green and blue images to YUV color space. Mostly inspired from: https://kornia.readthedocs.io/en/latest/_modules/kornia/color/yuv.html

    Parameters
    ----------
    image           : torch.tensor
                      Input image in RGB color space [k x 3 x m x n] or [3 x m x n].
    
    Returns
    -------
    image_yuv       : torch.tensor
                      Output image in YUV color space [k x 3 x m x n].
    """
    if len(image.shape) == 3:
       image = image.unsqueeze(0)
    red = image[:, 0, :, :]
    green = image[:, 1, :, :]
    blue  = image[:, 2, :, :]
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    u = -0.147 * red - 0.289 * green + 0.436 * blue
    v = 0.615 * red - 0.515 * green - 0.100 * blue 
    image_yuv = torch.stack([y, u, v], dim=0)
    return image_yuv
