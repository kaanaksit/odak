import odak.tools
import torch
import numpy as np_cpu


def resize(image, multiplier = 0.5, mode = 'nearest'):
    """
    Definition to resize an image.

    Parameters
    ----------
    image         : torch.tensor
                    Image with MxNx3 resolution.
    multiplier    : float
                    Multiplier used in resizing operation (e.g., 0.5 is half size in one axis).
    mode          : str
                    Mode to be used in scaling, nearest, bilinear, etc.

    Returns
    -------
    new_image     : torch.tensor
                    Resized image.

    """
    scale = torch.nn.Upsample(scale_factor = multiplier, mode = mode)
    new_image = torch.zeros((int(image.shape[0] * multiplier), int(image.shape[1] * multiplier), 3)).to(image.device)
    for i in range(3):
        cache = image[:,:,i].unsqueeze(0)
        cache = cache.unsqueeze(0)
        new_cache = scale(cache).unsqueeze(0)
        new_image[:,:,i] = new_cache.unsqueeze(0)
    return new_image


def load_image(fn, normalizeby = 0., torch_style = False):
    """
    Definition to load an image from a given location as a Numpy array.

    Parameters
    ----------
    fn           : str
                   Filename.
    normalized   : float
                   Value to to normalize images with. Default value of zero will lead to no normalization.
    torch_style  : bool
                   If set True, it will load an image mxnx3 as 3xmxn.


    Returns
    ----------
    image        :  ndarray
                    Image loaded as a Numpy array.

    """
    image = odak.tools.load_image(fn, normalizeby=normalizeby, torch_style=torch_style)
    image = torch.from_numpy(image).float()
    return image


def save_image(fn, img, cmin=0, cmax=255, color_depth=8):
    """
    Definition to save a Numpy array as an image.

    Parameters
    ----------
    fn           : str
                   Filename.
    img          : ndarray
                   A numpy array with NxMx3 or NxMx1 shapes.
    cmin         : int
                   Minimum value that will be interpreted as 0 level in the final image.
    cmax         : int
                   Maximum value that will be interpreted as 255 level in the final image.
    color_depth  : int
                   Color depth of an image. Default is eight.


    Returns
    ----------
    bool         :  bool
                    True if successful.

    """
    if len(img.shape) > 2 and torch.argmin(torch.tensor(img.shape)) == 0:
        new_img = torch.zeros(img.shape[1], img.shape[2], img.shape[0]).to(img.device)
        for i in range(img.shape[0]):
            new_img[:, :, i] = img[i].detach().clone()
        img = new_img.detach().clone()
    img = img.cpu().detach().numpy()
    return odak.tools.save_image(fn, img, cmin=cmin, cmax=cmax, color_depth=color_depth)
