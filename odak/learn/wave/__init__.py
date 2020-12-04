"""
``odak.learn.wave``
===================
Provides necessary definitions for neural networks and learning algorithms. The definitions are based on torch framework. Provides necessary definitions for merging geometric optics with wave theory and classical approaches in the wave theory as well. See "Introduction to Fourier Optcs" from Joseph Goodman for the theoratical explanation.

"""
from odak import np
import torch
from .classical import *


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
    phase = torch.angle(field)
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
    amplitude = torch.abs(field)
    return amplitude

def set_amplitude(field,amplitude):
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
    new_field = amplitude*torch.cos(phase)+1j*amplitude*torch.sin(phase)
    return new_field
