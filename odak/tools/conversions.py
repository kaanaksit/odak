from odak import np
import torch

def convert_to_torch(a):
    """
    A definition to convert Numpy/Cupy arrays to Torch.

    Parameters
    ----------
    a          : ndarray
                 Input Numpy or Cupy array.

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
    return c
