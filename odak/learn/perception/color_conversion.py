import torch


def rgb_2_ycrcb(image):
    """
    Converts an image from RGB colourspace to YCrCb colourspace.

    Parameters
    ----------
    image   : torch.tensor
              Input image. Should be an RGB floating-point image with values in the range [0, 1]. Should be in NCHW format [3 x m x n] or [k x 3 x m x n].

    Returns
    -------

    ycrcb   : torch.tensor
              Image converted to YCrCb colourspace [k x 3 m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
       image = image.unsqueeze(0)
    ycrcb = torch.zeros(image.size()).to(image.device)
    ycrcb[:, 0, :, :] = 0.299 * image[:, 0, :, :] + 0.587 * \
        image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
    ycrcb[:, 1, :, :] = 0.5 + 0.713 * (image[:, 0, :, :] - ycrcb[:, 0, :, :])
    ycrcb[:, 2, :, :] = 0.5 + 0.564 * (image[:, 2, :, :] - ycrcb[:, 0, :, :])
    return ycrcb


def ycrcb_2_rgb(image):
    """
    Converts an image from YCrCb colourspace to RGB colourspace.

    Parameters
    ----------
    image   : torch.tensor
              Input image. Should be a YCrCb floating-point image with values in the range [0, 1]. Should be in NCHW format [3 x m x n] or [k x 3 x m x n].

    Returns
    -------
    rgb     : torch.tensor
              Image converted to RGB colourspace [k x 3 m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
       image = image.unsqueeze(0)
    rgb = torch.zeros(image.size(), device=image.device)
    rgb[:, 0, :, :] = image[:, 0, :, :] + 1.403 * (image[:, 1, :, :] - 0.5)
    rgb[:, 1, :, :] = image[:, 0, :, :] - 0.714 * \
        (image[:, 1, :, :] - 0.5) - 0.344 * (image[:, 2, :, :] - 0.5)
    rgb[:, 2, :, :] = image[:, 0, :, :] + 1.773 * (image[:, 2, :, :] - 0.5)
    return rgb


import torch


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
                      Output image in YUV color space [k x 3 x m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
       image = image.unsqueeze(0)
    red = image[:, 0, :, :]
    green = image[:, 1, :, :]
    blue  = image[:, 2, :, :]
    image_yuv = torch.zeros_like(image)
    image_yuv[:, 0, :, :] = 0.299 * red + 0.587 * green + 0.114 * blue
    image_yuv[:, 1, :, :] = -0.147 * red - 0.289 * green + 0.436 * blue
    image_yuv[:, 2, :, :] = 0.615 * red - 0.515 * green - 0.100 * blue
    return image_yuv



def convert_yuv_to_rgb(image):
    """
    Definition to convert YUV images to RGB color space. Mostly inspired from: https://kornia.readthedocs.io/en/latest/_modules/kornia/color/yuv.html

    Parameters
    ----------
    image           : torch.tensor
                      Input image in YUV color space [k x 3 x m x n] or [3 x m x n].
    
    Returns
    -------
    image_yuv       : torch.tensor
                      Output image in RGB color space [k x 3 x m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
       image = image.unsqueeze(0)
    y = image[:, 0, :, :]
    u = image[:, 1, :, :]
    v = image[:, 2, :, :]
    image_rgb = torch.zeros_like(image)
    image_rgb[:, 0, :, :] = y + 1.14 * v
    image_rgb[:, 1, :, :] = y + -0.396 * u - 0.581 * v
    image_rgb[:, 2, :, :] = y + 2.029 * u
    return image_rgb


def convert_rgb_to_linear_rgb(image, threshold = 0.0031308):
    """
    Definition to convert RGB images to linear RGB color space. Mostly inspired from: https://kornia.readthedocs.io/en/latest/_modules/kornia/color/rgb.html#linear_rgb_to_rgb

    Parameters
    ----------
    image           : torch.tensor
                      Input image in RGB color space [k x 3 x m x n] or [3 x m x n]. Image(s) must be normalized between zero and one.
    threshold       : float
                      Threshold used in calculations.

    Returns
    -------
    image_linear    : torch.tensor
                      Output image in linear RGB color space [k x 3 x m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    threshold = 0.0031308
    image_linear =  torch.where(image > threshold, 1.055 * torch.pow(image.clamp(min=threshold), 1 / 2.4) - 0.055, 12.92 * image)
    return image_linear
