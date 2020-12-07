"""
``odak.learn``
===================
Provides necessary definitions for neural networks and learning algorithms. The definitions are based on torch framework.
"""
from odak import np
import torch
from .classical import *


def complex_to_polar(field):
    """Converts the complex-value tensor to polar"""
    return torch.stack([torch.abs(field), torch.angle(field)], dim=-1)

def complex_to_rect(field):
    """Converts the complex-value tensor to complex representation"""
    return torch.stack([field.real, field.imag], dim=-1)

def rect_to_polar(real, imag):
    """Converts the rectangular complex representation to polar"""
    mag = torch.pow(real**2 + imag**2, 0.5)
    ang = torch.atan2(imag, real)
    return mag, ang

def polar_to_rect(mag, ang):
    """Converts the polar complex representation to rectangular"""
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)
    return real, imag

def polar_exp(field):
    mag, phase = field.split(1, -1)
    result = torch.exp(mag)* torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
    return result


def complex_div(t1, t2):
    """divide two complex valued tensors element-wise. the two last dimensions are
    assumed to be the real and imaginary part

    complex division: (a+bi) / (c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2) i
    """
    assert t1.shape[-1] == 2, "the last dim of input tensor t1 must be 2"
    assert t2.shape[-1] == 2, "the last dim of input tensor t2 must be 2"

    # real and imaginary parts of first tensor
    (a, b) = t1.split(1, -1)
    # real and imaginary parts of second tensor
    (c, d) = t2.split(1, -1)

    # get magnitude
    mag = torch.mul(c, c) + torch.mul(d, d)
    # multiply out
    return torch.cat(((a * c + b * d) / mag, (b * c - a * d) / mag), -1)

def complex_mul(t1, t2):
    """multiply two complex valued tensors element-wise. the two last dimensions are
    assumed to be the real and imaginary part

    complex multiplication: (a+bi)(c+di) = (ac-bd) + (bc+ad)i
    """

    assert t1.shape[-1] == 2, "the last dim of input tensor t1 must be 2"
    assert t2.shape[-1] == 2, "the last dim of input tensor t2 must be 2"

    # real and imaginary parts of first tensor
    a, b = t1.split(1, -1)
    # real and imaginary parts of second tensor
    c, d = t2.split(1, -1)

    # multiply out
    return torch.cat((a * c - b * d, b * c + a * d), -1)


def calculate_phase(field,deg=False):
    """ 
    Definition to calculate phase of a single or multiple given electric field(s).

    Parameters
    ----------
    field        : torch.cfloat
                   Electric fields or an electric field.
    deg          : bool
                   If set True, the angles will be returned in degrees.

    Returns
    ----------
    phase        : torch.float
                   Phase or phases of electric field(s) in radians.
    """
    if field.size(-1) != 2:
        phase = torch.angle(field)
    else:
        _, phase = rect_to_polar(field[...,0], field[...,1])
    if deg == True:
        phase *= 180./np.pi
    return phase

def calculate_amplitude(field):
    """ 
    Definition to calculate amplitude of a single or multiple given electric field(s).

    Parameters
    ----------
    field        : torch.cfloat
                   Electric fields or an electric field.

    Returns
    ----------
    amplitude    : torch.float
                   Amplitude or amplitudes of electric field(s).
    """
    if field.size(-1) != 2:
        amplitude = torch.abs(field)
    else:
        amplitude, _ = rect_to_polar(field[...,0], field[...,1])
    return amplitude

def set_amplitude(field, amplitude):
    """
    Definition to keep phase as is and change the amplitude of a given field.

    Parameters
    ----------
    field        : torch.cfloat
                   Complex field.
    amplitude    : torch.cfloat or torch.float
                   Amplitudes.

    Returns
    ----------
    new_field    : torch.cfloat
                   Complex field.
    """
    amplitude = calculate_amplitude(amplitude)
    phase     = calculate_phase(field)
    print(field.dtype)
    if field.dtype == torch.cfloat or field.dtype == torch.complex64 or field.dtype==torch.complex128:
        new_field = amplitude*torch.cos(phase)+1j*amplitude*torch.sin(phase)
    else:
        new_field = torch.stack([amplitude*torch.cos(phase), amplitude * torch.sin(phase)], dim=-1)
    return new_field