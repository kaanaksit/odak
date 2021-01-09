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

def generate_complex_field(amplitude,phase):
    """
    Definition to generate a complex field with a given amplitude and phase.

    Parameters
    ----------
    amplitude         : ndarray
                        Amplitude of the field.
    phase             : ndarray
                        Phase of the field.

    Returns
    ----------
    field             : ndarray
                        Complex field.
    """
    if type(phase) == 'torch.Tensor':
        phase     = torch.FloatTensor(phase)
    elif type(phase) == type([]):
        phase     = torch.FloatTensor([phase])
    if type(amplitude) == 'torch.Tensor':
        amplitude = torch.FloatTensor(amplitude)
    elif type(amplitude) == type([]):
        amplitude = torch.FloatTensor([amplitude])
    field     = amplitude*torch.cos(phase)+1j*amplitude*torch.sin(phase)
    return field

def produce_phase_only_slm_pattern(hologram,slm_range):
    """
    Definition for producing a pattern for a phase only Spatial Light Modulator (SLM) using a given field.

    Parameters
    ==========
    hologram           : torch.cfloat
                         Input holographic field.
    slm_range          : float
                         Range of the phase only SLM in radians for a working wavelength (i.e. two pi). See odak.wave.adjust_phase_only_slm_range() for more.
    filename           : str
                         Optional variable, if provided the patterns will be save to given location.

    Returns
    ==========
    pattern            : torch.cfloat
                         Adjusted phase only pattern.
    """
    hologram_phase                            = calculate_phase(hologram) % (2*np.pi)
    hologram_phase[hologram_phase>slm_range]  = slm_range
    hologram_phase                           /= slm_range
    hologram_phase                           *= 255
    hologram_phase                            = hologram_phase.int()
    hologram_phase                            = hologram_phase.float()
    hologram_phase                           *= slm_range/255.
    return torch.cos(hologram_phase)+1j*torch.sin(hologram_phase)
