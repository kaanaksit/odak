import sys
import os
import odak
import numpy as np
import torch


def test():
    wavelength = 532e-9 # (1)
    pixel_pitch = 8e-6 # (2)
    distance = 0.5e-2 # (3)
    propagation_type = 'Angular Spectrum' # (4)
    k = odak.learn.wave.wavenumber(wavelength) # (5)


    amplitude = torch.zeros(500, 500)
    amplitude[200:300, 200:300 ] = 1. # (5)
    phase = torch.randn_like(amplitude) * 2 * odak.pi # (6)
    hologram = odak.learn.wave.generate_complex_field(amplitude, phase) # (7)


    image_plane = odak.learn.wave.propagate_beam(
                                                 hologram,
                                                 k,
                                                 distance,
                                                 pixel_pitch,
                                                 wavelength,
                                                 propagation_type,
                                                 zero_padding = [True, False, True] # (8)
                                                ) # (9)

    image_intensity = odak.learn.wave.calculate_amplitude(image_plane) ** 2 # (10)
    hologram_intensity = amplitude ** 2

    odak.learn.tools.save_image(
                                'image_intensity.png', 
                                image_intensity, 
                                cmin = 0., 
                                cmax = 1.
                               ) # (11)
    odak.learn.tools.save_image(
                                'hologram_intensity.png', 
                                hologram_intensity, 
                                cmin = 0., 
                                cmax = 1.
                               ) # (12)
    assert True == True


if __name__ == '__main__':
    sys.exit(test()) 















