"""
``odak.jones``
===================
Provides necessary definitions for jones calculus. See "Introduction to Fourier Optics" by Joseph Goodman.
"""
import numpy as np


def electricfield(px, py):
    """
    Definition to create an electric field vector (polarization vector).

    Parameters
    ----------
    px           : float
                   Amplitude of the electric field along X axis.
    py           : float
                   Amplitude of the electric field along Y axis.

    Returns
    ----------
    field        : ndarray
                   An electric field vector (polarization vector).
    """
    field = np.array([[px], [py]])
    return field


def linearpolarizer(field, rotation=0):
    """
    Definition that represents a linear polarizer.

    Parameters
    ----------
    field        : ndarray
                   Polarization vector of an input beam.
    rotation     : float
                   Represents rotation of the polarizer along propagation direction in angles (couter-clockwise).

    Returns
    ----------
    result       : ndarray
                   Polarization vector of an output beam.
    """
    rotation = np.radians(rotation)
    rotmat = np.array(
        [
            [float(np.cos(rotation)), float(np.sin(rotation))],
            [float(-np.sin(rotation)),
             float(np.cos(rotation))]
        ]
    )
    linearpolarizer = np.array([[1, 0], [0, 0]])
    linearpolarizer = np.dot(
        rotmat.transpose(), np.dot(linearpolarizer, rotmat))
    result = np.dot(linearpolarizer, field)
    return result
