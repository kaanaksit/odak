import numpy as np
import torch


def convert_to_torch(a, grad=True):
    """
    A definition to convert Numpy arrays to Torch.

    Parameters
    ----------
    a          : ndarray
                 Input Numpy array.
    grad       : bool
                 Set if the converted array requires gradient.

    Returns
    ----------
    c          : torch.Tensor
                 Converted array.
    """
    b = np.copy(a)
    c = torch.from_numpy(b)
    c.requires_grad_(grad)
    return c


def convert_to_numpy(a):
    """
    A definition to convert Torch to Numpy.

    Parameters
    ----------
    a          : torch.Tensor
                 Input Torch array.

    Returns
    ----------
    b          : numpy.ndarray
                 Converted array.
    """
    b = a.to('cpu').detach().numpy()
    return b
