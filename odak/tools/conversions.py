from odak import np
import torch

def convert_to_torch(a,grad=True):
    """
    A definition to convert Numpy/Cupy arrays to Torch.

    Parameters
    ----------
    a          : ndarray
                 Input Numpy or Cupy array.
    grad       : bool
                 Set if the converted array requires gradient.

    Returns
    ----------
    c          : torch.Tensor
                 Converted array.
    """
    if np.__name__ == 'cupy':
        b = np.asnumpy(a)
    else:
        b = np.copy(a)
    c = torch.from_numpy(b)
    c.requires_grad_(grad)
    return c

def convert_to_numpy(a):
    """
    A definition to convert Torch/Cupy to Numpy.

    Parameters
    ----------
    a          : torch.Tensor or cupy.ndarray
                 Input Cupy or Torch array.

    Returns
    ----------
    b          : numpy.ndarray
                 Converted array.
    """
    if str(type(a)).find('torch') > 0:
        b = a.to('cpu').detach().numpy()
    elif str(type(a)).find('cupy') > 0:
        b = np.asnumpy(a)
    return b
