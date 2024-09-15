import sys
import os
import odak
import numpy as np
import torch


def test(output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    wavelength = 532e-9 # (1)
    pixel_pitch = 8e-6 # (2)
    distance = 0.5e-2 # (3)
    propagation_types = ['Angular Spectrum', 'Bandlimited Angular Spectrum', 'Transfer Function Fresnel'] # (4)
    k = odak.learn.wave.wavenumber(wavelength) # (5)

    
    amplitude = torch.zeros(500, 500)
    amplitude[200:300, 200:300 ] = 1. # (5)
    phase = torch.randn_like(amplitude) * 2 * odak.pi # (6)
    hologram = odak.learn.wave.generate_complex_field(amplitude, phase) # (7)

    for propagation_type in propagation_types:
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
                                    '{}/image_intensity_{}.png'.format(output_directory, propagation_type.replace(' ', '_')), 
                                    image_intensity, 
                                    cmin = 0., 
                                    cmax = image_intensity.max()
                                ) # (11)
        odak.learn.tools.save_image(
                                    '{}/hologram_intensity_{}.png'.format(output_directory, propagation_type.replace(' ', '_')), 
                                    hologram_intensity, 
                                    cmin = 0., 
                                    cmax = 1.
                                ) # (12)
    assert True == True


if __name__ == '__main__':
    sys.exit(test()) 















