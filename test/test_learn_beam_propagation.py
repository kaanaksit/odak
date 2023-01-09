import sys
import os
import odak
import numpy as np
from odak.wave import wavenumber, propagate_beam
from odak.learn.wave import propagate_beam as propagate_beam_torch
from odak.tools import zero_pad
import torch


def test():
    wavelength = 0.5*pow(10, -6)
    pixeltom = 6*pow(10, -6)
    distance = 0.2
    propagation_types = ['TR Fresnel', 'Angular Spectrum', 'Bandlimited Angular Spectrum']
    k = wavenumber(wavelength)
    sample_field = np.zeros((500, 500), dtype=np.complex64)
    sample_field[
        240:260,
        240:260
    ] = 1000
    random_phase = np.pi*np.random.random(sample_field.shape)
    sample_field = sample_field * \
        np.cos(random_phase)+1j*sample_field*np.sin(random_phase)
    sample_field = zero_pad(sample_field)

    sample_field_torch = torch.from_numpy(sample_field)

    for i in range(len(propagation_types)):
        propagation_type = propagation_types[i]
        # Propagate and reconstruct using torch.
        hologram_torch = propagate_beam_torch(
            sample_field_torch,
            k,
            distance,
            pixeltom,
            wavelength,
            propagation_type
        )
        reconstruction_torch = propagate_beam_torch(
            hologram_torch,
            k,
            -distance,
            pixeltom,
            wavelength,
            propagation_type
        )

    assert True == True


if __name__ == '__main__':
    sys.exit(test())
