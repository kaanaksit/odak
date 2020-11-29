from odak import np
import torch
import math

def set_amplitude(field,amplitude):
    """
    Definition to keep phase as is and change the amplitude of a given field.
    Parameters
    ----------
    field        : np.complex64
                   Complex field.
    amplitude    : np.array or np.complex64
                   Amplitudes.
    Returns
    ----------
    new_field    : np.complex64
                   Complex field.
    """
    amplitude = calculate_amplitude(amplitude)
    phase     = calculate_phase(field)
    new_field = amplitude*np.cos(phase)+1j*amplitude*np.sin(phase)
    return new_field

def calculate_amplitude(field):
    """ 
    Definition to calculate amplitude of a single or multiple given electric field(s).
    Parameters
    ----------
    field        : ndarray.complex or complex
                   Electric fields or an electric field.
    Returns
    ----------
    amplitude    : float
                   Amplitude or amplitudes of electric field(s).
    """
    amplitude = np.abs(field)
    return amplitude

def calculate_intensity(field):
    """
    Definition to calculate intensity of a single or multiple given electric field(s).
    Parameters
    ----------
    field        : ndarray.complex or complex
                   Electric fields or an electric field.
    Returns
    ----------
    intensity    : float
                   Intensity or intensities of electric field(s).
    """
    intensity = np.abs(field)**2
    return intensity

## The following functions are revised from https://github.com/computational-imaging/neural-holography
def ifftshift(tensor):
    """ifftshift for tensors of dimensions [height, width]
    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, -math.floor(size[0] / 2.0), 0)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[1] / 2.0), 1)
    return tensor_shifted


def fftshift(tensor):
    """fftshift for tensors of dimensions [height, width]
    shifts the width and heights
    """
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[0] / 2.0), 0)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[1] / 2.0), 1)
    return tensor_shifted

def roll_torch(tensor, shift, axis):
    """implements numpy roll() or Matlab circshift() functions for tensors"""
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)
