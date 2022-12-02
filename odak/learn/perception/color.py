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
