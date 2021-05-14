from PIL import Image
import odak.tools
import torch
import numpy as np_cpu

def load_image(fn):
    """
    Definition to load an image from a given location as a Numpy array.

    Parameters
    ----------
    fn           : str
                   Filename.

    Returns
    ----------
    image        :  ndarray
                    Image loaded as a Numpy array.

    """
    image = Image.open(fn)
    image = np_cpu.array(image)
    image = torch.from_numpy(image)
    return image

def save_image(fn,img,cmin=0,cmax=255):
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

    Returns
    ----------
    bool         :  bool
                    True if successful.

    """
    img = img.cpu().detach().numpy()
    return odak.tools.save_image(fn,img,cmin,cmax)
