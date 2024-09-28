import sys
import os
import odak
import numpy as np
import torch


def test(device = torch.device('cpu'), output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    wavelength = 532e-9
    pixel_pitch = 3.74e-6
    distance = 1e-3
    aperture_samples = [50, 50, 1, 1]
    propagation_types = [
                         'Seperable Impulse Response Fresnel',
                         'Impulse Response Fresnel',
                         'Transfer Function Fresnel',
                         'Angular Spectrum',
                         'Bandlimited Angular Spectrum',
                         'Incoherent Angular Spectrum',
                        ]
    k = odak.learn.wave.wavenumber(wavelength)

    hologram = odak.learn.tools.load_image('test/data/sample_hologram.png', normalizeby = 255., torch_style = True).to(device) * 2 * odak.pi
    hologram = odak.learn.wave.generate_complex_field(1., hologram[1, 0:500, 0: 500])

    for propagation_type in propagation_types:
        propagator = odak.learn.wave.propagator(
                                                wavelengths = [wavelength,],
                                                pixel_pitch = pixel_pitch,
                                                resolution = [hologram.shape[-2], hologram.shape[-1]],
                                                aperture_size = hologram.shape[-2],
                                                aperture_samples = aperture_samples,
                                                number_of_frames = 1,
                                                number_of_depth_layers = 1,
                                                volume_depth = 1e-3,
                                                image_location_offset = distance,
                                                propagation_type = propagation_type,
                                                propagator_type = 'forward',
                                                resolution_factor = 1,
                                                method = 'conventional',
                                                device = device
                                               )
        image_plane = propagator.reconstruct(
                                             hologram_phases = odak.learn.wave.calculate_phase(hologram).unsqueeze(0),
                                             amplitude = odak.learn.wave.calculate_amplitude(hologram).unsqueeze(0),
                                             get_complex = True
                                            )[0, 0, 0]
        image_intensity = odak.learn.wave.calculate_amplitude(image_plane) ** 2
        image_phase = odak.learn.wave.calculate_phase(image_plane) % (2 * odak.pi)
        odak.learn.tools.save_image(
                                    '{}/comparison_phase_{}.png'.format(output_directory, propagation_type.replace(' ', '_')), 
                                    image_phase, 
                                    cmin = 0., 
                                    cmax = 2. * odak.pi
                                   ) 
        odak.learn.tools.save_image(
                                    '{}/comparison_intensity_{}.png'.format(output_directory, propagation_type.replace(' ', '_')), 
                                    image_intensity, 
                                    cmin = 0., 
                                    cmax = image_intensity.max()
                                   ) 
    hologram_intensity = odak.learn.wave.calculate_amplitude(hologram) ** 2
    odak.learn.tools.save_image(
                                '{}/hologram_intensity.png'.format(output_directory), 
                                hologram_intensity, 
                                cmin = 0., 
                                cmax = 1.
                               )
    assert True == True


if __name__ == '__main__':
    sys.exit(test()) 















