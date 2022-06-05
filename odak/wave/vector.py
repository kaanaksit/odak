import numpy as np
from ..tools import distance_between_two_points
from .utils import calculate_amplitude, calculate_phase


def propagate_field(points0, points1, field0, wave_number, direction=1):
    """
    Definition to propagate a field from points to an another points in space: propagate a given array of spherical sources to given set of points in space.

    Parameters
    ----------
    points0       : ndarray
                    Start points (i.e. odak.tools.grid_sample).
    points1       : ndarray
                    End points (ie. odak.tools.grid_sample).
    field0        : ndarray
                    Field for given starting points.
    wave_number   : float
                    Wave number of a wave, see odak.wave.wavenumber for more.
    direction     : float
                    For propagating in forward direction set as 1, otherwise -1.

    Returns
    -------
    field1        : ndarray
                    Field for given end points.
    """
    field1 = np.zeros(points1.shape[0], dtype=np.complex64)
    for point_id in range(points0.shape[0]):
        point = points0[point_id]
        distances = distance_between_two_points(
            point,
            points1
        )
        field1 += electric_field_per_plane_wave(
            calculate_amplitude(field0[point_id]),
            distances*direction,
            wave_number,
            phase=calculate_phase(field0[point_id])
        )
    return field1


def propagate_plane_waves(field, opd, k, w=0, t=0):
    """
    Definition to propagate a field representing a plane wave at a particular distance and time.

    Parameters
    ----------
    field        : complex
                   Complex field.
    opd          : float
                   Optical path difference in mm.
    k            : float
                   Wave number of a wave, see odak.wave.parameters.wavenumber for more.
    w            : float
                   Rotation speed of a wave, see odak.wave.parameters.rotationspeed for more.
    t            : float
                   Time in seconds.

    Returns
    -------
    new_field     : complex
                    A complex number that provides the resultant field in the complex form A*e^(j(wt+phi)).
    """
    new_field = field*np.exp(1j*(-w*t+opd*k))/opd**2
    return new_field


def electric_field_per_plane_wave(amplitude, opd, k, phase=0, w=0, t=0):
    """
    Definition to return state of a plane wave at a particular distance and time.

    Parameters
    ----------
    amplitude    : float
                   Amplitude of a wave.
    opd          : float
                   Optical path difference in mm.
    k            : float
                   Wave number of a wave, see odak.wave.parameters.wavenumber for more.
    phase        : float
                   Initial phase of a wave.
    w            : float
                   Rotation speed of a wave, see odak.wave.parameters.rotationspeed for more.
    t            : float
                   Time in seconds.

    Returns
    -------
    field        : complex
                   A complex number that provides the resultant field in the complex form A*e^(j(wt+phi)).
    """
    field = amplitude*np.exp(1j*(-w*t+opd*k+phase))/opd**2
    return field
