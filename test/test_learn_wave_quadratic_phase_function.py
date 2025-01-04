import sys
import os
import odak
import numpy as np
import torch


def test(
         resolution = [250, 250],
         wavelengths = [639e-9, 515e-9, 473e-9],
         pixel_pitch = 3.74e-6,
         number_of_frames = 3,
         number_of_depth_layers = 10,
         volume_depth = 10e-3,
         image_location_offset = 3e-2,
         propagation_type = 'Bandlimited Angular Spectrum',
         propagator_type = 'forward',
         laser_channel_power = torch.tensor([
                                             [1., 0., 0.],
                                             [0., 1., 0.],
                                             [0., 0., 1.],
                                            ]),
         lens_focus = 3e-2,
         aperture = None,
         aperture_size = None,
         method = 'conventional',
         device = torch.device('cpu'),
         output_directory = 'test_output'
        ):
    odak.tools.check_directory(output_directory)
    hologram_phases = torch.ones(number_of_frames, resolution[0], resolution[1], device = device)
    for frame_id in range(number_of_frames):
        wavelength = wavelengths[frame_id]
        k = odak.learn.wave.wavenumber(wavelength)
        lens_complex =  odak.learn.wave.quadratic_phase_function(
                                                                 nx = resolution[0],
                                                                 ny = resolution[1],
                                                                 k = k,
                                                                 focal = lens_focus,
                                                                 dx = pixel_pitch
                                                                )
        lens_phase = odak.learn.wave.calculate_phase(lens_complex).to(device).unsqueeze(0) % (2. * torch.pi) 
        hologram_phases[frame_id] = lens_phase
    odak.learn.tools.save_image(
                                '{}/lens_phase.png'.format(output_directory),
                                hologram_phases,
                                cmin = 0., 
                                cmax = 2. * torch.pi
                               )
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
    for frame_id in range(reconstruction_intensities.shape[0]):
        for depth_id in range(reconstruction_intensities.shape[1]):
            reconstruction_intensity = reconstruction_intensities[frame_id, depth_id]
            odak.learn.tools.save_image(
                                        '{}/lens_reconstruction_image_{:03d}_{:03d}.png'.format(output_directory, frame_id, depth_id),
                                        reconstruction_intensity,
                                        cmin = 0.,
                                        cmax = reconstruction_intensities.max()
                                       )
    assert True == True


if __name__ == '__main__':
    sys.exit(test()) 
