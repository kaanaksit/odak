import sys
import os
import odak
import numpy as np
import torch


def test(output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    resolution = [2400, 4094]
    wavelengths = [639e-9, 515e-9, 473e-9]
    pixel_pitch = 3.74e-6
    number_of_frames = 3
    number_of_depth_layers = 3
    volume_depth = 5e-3
    image_location_offset = 0.
    propagation_type = 'Bandlimited Angular Spectrum'
    propagator_type = 'forward'
    laser_channel_power = None
    aperture = None
    aperture_size = 1800
    method = 'conventional'
    device = torch.device('cpu')
    hologram_phases_filename = './test/data/sample_hologram.png'
    hologram_phases = odak.learn.tools.load_image(
                                                  hologram_phases_filename,
                                                  normalizeby = 255.,
                                                  torch_style = True
                                                 ).to(device) * odak.pi * 2
    propagator = odak.learn.wave.propagator(
                                            resolution = resolution,
                                            wavelengths = wavelengths,
                                            pixel_pitch = pixel_pitch,
                                            number_of_frames = number_of_frames,
                                            number_of_depth_layers = number_of_depth_layers,
                                            volume_depth = volume_depth,
                                            image_location_offset = image_location_offset,
                                            propagation_type = propagation_type,
                                            propagator_type = propagator_type,
                                            laser_channel_power = laser_channel_power,
                                            aperture_size = aperture_size,
                                            aperture = aperture,
                                            method = method,
                                            device = device
                                           )
    reconstruction_intensities = propagator.reconstruct(hologram_phases, amplitude = None)
    reconstruction_intensities = torch.sum(reconstruction_intensities, axis = 0)
    for depth_id, reconstruction_intensity in enumerate(reconstruction_intensities):
        odak.learn.tools.save_image(
                                    '{}/reconstruction_image_{:03d}.png'.format(output_directory, depth_id),
                                    reconstruction_intensity,
                                    cmin = 0.,
                                    cmax = 1.
                                   )
    assert True == True


if __name__ == '__main__':
    sys.exit(test()) 















