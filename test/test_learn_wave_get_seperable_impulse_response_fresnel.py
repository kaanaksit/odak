import odak
import torch
import sys

from tqdm import tqdm


def get_1D_kernels(
                   resolution, 
                   pixel_pitches,
                   wavelengths,
                   distances,
                   scale = 1,
                   aperture_samples = [50, 50, 5, 5],
                   device = torch.device('cpu')
                  ):
    kernels_x = torch.zeros(
                            len(wavelengths),
                            len(distances),
                            len(pixel_pitches),
                            resolution[0] * scale,
                            dtype = torch.complex64,
                            device = device
                           )
    kernels_y = torch.zeros(
                            len(wavelengths),
                            len(distances),
                            len(pixel_pitches),
                            resolution[1] * scale,
                            dtype = torch.complex64,
                            device = device
                           )
    kernels = torch.zeros(
                          len(wavelengths),
                          len(distances),
                          len(pixel_pitches),
                          resolution[0] * scale,
                          resolution[1] * scale,
                          dtype = torch.complex64,
                          device = device
                         )
    for wavelength_id, wavelength in enumerate(wavelengths):
        for dx_id, dx in enumerate(pixel_pitches):
            for distance_id in tqdm(range(len(distances))):
                distance = distances[distance_id]
                _, kernel, kernel_x, kernel_y = odak.learn.wave.get_seperable_impulse_response_fresnel_kernel(
                                                                                                       nu = resolution[0],
                                                                                                       nv = resolution[1],
                                                                                                       dx = dx,
                                                                                                       wavelength = wavelength,
                                                                                                       distance = distance,
                                                                                                       scale = scale,
                                                                                                       aperture_samples = aperture_samples,
                                                                                                       device = device
                                                                                                      )
                kernels[
                        wavelength_id,
                        distance_id,
                        dx_id
                       ] = kernel.detach().clone()
                kernels_x[
                          wavelength_id,
                          distance_id ,
                          dx_id
                         ] = kernel_x.detach().clone()
                kernels_y[
                          wavelength_id,
                          distance_id ,
                          dx_id
                         ] = kernel_y.detach().clone()
    return kernels, kernels_x, kernels_y


def main(
         wavelengths = [515e-9],
         distance_range = [1e-4, 10.0e-3],
         distance_no = 1,
         pixel_pitches = [3.74e-6],
         resolution = [512, 512],
         resolution_factor = 1,
         propagation_type = 'Impulse Response Fresnel',
         samples = [50, 50, 1, 1],
         device = torch.device('cpu'),
         output_directory = 'output'
        ):
    odak.tools.check_directory(output_directory)
    distances = torch.linspace(distance_range[0], distance_range[1], distance_no)
    light_kernels, light_kernels_x, light_kernels_y = get_1D_kernels(
                                                                     resolution = resolution,
                                                                     pixel_pitches = pixel_pitches,
                                                                     wavelengths = wavelengths,
                                                                     distances = distances,
                                                                     scale = resolution_factor,
                                                                     aperture_samples = samples,
                                                                     device = device
                                                                    )
    save(
         light_kernels_x,
         directory = '{}/vectorized/'.format(output_directory)
        )
    save_psfs(
              light_kernels,
              directory = '{}/vectorized/psfs/'.format(output_directory)
             )
    assert True == True


def save_psfs(kernels, directory):
    odak.tools.check_directory(directory)
    for wavelength_id in range(kernels.shape[0]):
        for distance_id in range(kernels.shape[1]):
            for pixel_pitch_id in range(kernels.shape[2]):
                kernel = kernels[wavelength_id, distance_id, pixel_pitch_id]
                kernel_amplitude = odak.learn.wave.calculate_amplitude(kernel)
                kernel_amplitude = kernel_amplitude / kernel_amplitude.max()
                kernel_intensity = kernel_amplitude ** 2
                kernel_phase = odak.learn.wave.calculate_phase(kernel) % (2 * torch.pi)
                kernel_weighted = kernel_amplitude * kernel_phase
                odak.learn.tools.save_image(
                                           '{}/intensity_w{:03d}_d{:03d}_p{:03d}.png'.format(
                                                                                             directory,
                                                                                             wavelength_id,
                                                                                             distance_id,
                                                                                             pixel_pitch_id,
                                                                                            ),
                                            kernel_intensity,
                                            cmin = 0.,
                                            cmax = kernel_intensity.max()
                                           )
                odak.learn.tools.save_image(
                                           '{}/amplitude_w{:03d}_d{:03d}_p{:03d}.png'.format(
                                                                                             directory,
                                                                                             wavelength_id,
                                                                                             distance_id,
                                                                                             pixel_pitch_id,
                                                                                            ),
                                            kernel_amplitude,
                                            cmin = 0.,
                                            cmax = kernel_amplitude.max()
                                           )
                odak.learn.tools.save_image(
                                            '{}/phase_w{:03d}_d{:03d}_p{:03d}.png'.format(
                                                                                          directory,
                                                                                          wavelength_id,
                                                                                          distance_id,
                                                                                          pixel_pitch_id,
                                                                                         ),
                                            kernel_phase,
                                            cmin = 0.,
                                            cmax = 2 * torch.pi
                                           )
                odak.learn.tools.save_image(
                                            '{}/weighted_w{:03d}_d{:03d}_p{:03d}.png'.format(
                                                                                             directory,
                                                                                             wavelength_id,
                                                                                             distance_id,
                                                                                             pixel_pitch_id,
                                                                                             ),
                                            kernel_weighted,
                                            cmin = 0.,
                                            cmax = kernel_weighted.max()
                                           )
    return True

def save(kernels, directory):
    odak.tools.check_directory(directory)
    for wavelength_id in range(kernels.shape[0]):
        for pixel_pitch_id in range(kernels.shape[2]):
            kernel = kernels[wavelength_id, :, pixel_pitch_id]
            kernel_amplitude = odak.learn.wave.calculate_amplitude(kernel)
            kernel_amplitude = kernel_amplitude / kernel_amplitude.max()
            kernel_phase = odak.learn.wave.calculate_phase(kernel) % (2 * torch.pi)
            kernel_weighted = kernel_amplitude * kernel_phase
            odak.learn.tools.save_image(
                                        '{}/amplitude_w{:03}_p{:03d}.png'.format(
                                                                                 directory,
                                                                                 wavelength_id,
                                                                                 pixel_pitch_id,
                                                                                ),
                                        kernel_amplitude,
                                        cmin = 0.,
                                        cmax = kernel_amplitude.max()
                                       )
            odak.learn.tools.save_image(
                                        '{}/phase_w{:03}_p{:03d}.png'.format(
                                                                             directory,
                                                                             wavelength_id,
                                                                             pixel_pitch_id,
                                                                            ),
                                        kernel_phase,
                                        cmin = 0.,
                                        cmax = 2 * torch.pi
                                       )
            odak.learn.tools.save_image(
                                        '{}/weighted_w{:03}_p{:03d}.png'.format(
                                                                                directory,
                                                                                wavelength_id,
                                                                                pixel_pitch_id,
                                                                               ),
                                        kernel_weighted,
                                        cmin = 0.,
                                        cmax = kernel_weighted.max()
                                       )
    return True


if __name__ == '__main__':
    sys.exit(main())
