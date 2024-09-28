import odak
import torch
import sys

from tqdm import tqdm


def get_target_plane_points(
                            resolution = [50, 50],
                            resolution_factor = 1,
                            z = 1e-3,
                            pixel_pitch = 3.74e-6,
                            device = torch.device('cpu')
                           ):
    wx = resolution[0] * pixel_pitch
    wy = resolution[1] * pixel_pitch
    x = torch.linspace(-wx / 2., wx / 2., resolution[0] * resolution_factor, device = device)
    y = torch.linspace(-wy / 2., wy / 2., resolution[1] * resolution_factor, device = device)
    X, Y = torch.meshgrid(x, y, indexing = 'ij')
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    Z = torch.ones_like(X) * z
    target_plane_points = torch.cat((X, Y, Z), axis = 1)
    return target_plane_points


def get_aperture_points(
                        aperture_pattern,
                        z = 0.,
                        aperture_phase = None,
                        dimensions = [3.74e-6, 3.74e-6],
                        device = torch.device('cpu')
                       ):
    x = torch.linspace(- dimensions[0] / 2., dimensions[0] / 2., aperture_pattern.shape[-2], device = device)
    y = torch.linspace(- dimensions[1] / 2., dimensions[1] / 2., aperture_pattern.shape[-1], device = device)
    X, Y = torch.meshgrid(x, y, indexing = 'ij')
    X = X[aperture_pattern > 0.]
    Y = Y[aperture_pattern > 0.]
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    Z = torch.ones_like(X) * z
    aperture_points = torch.cat((X, Y, Z), axis = 1)
    if isinstance(aperture_phase, type(None)):
        aperture_phase = torch.zeros_like(aperture_pattern)
    aperture_field = odak.learn.wave.generate_complex_field(aperture_pattern, aperture_phase)
    aperture_field = aperture_field[aperture_pattern > 0.]
    aperture_field = aperture_field.reshape(1, -1)
    return aperture_points, aperture_field


def main(
         wavelength = 515e-9,
         distance = 0.5e-3,
         pixel_pitch = 3.74e-6,
         resolution = [512, 512],
         resolution_factor = 1,
         randomization = False,
         nx = 1, ny = 1,
         aperture_pattern_filename = './test/data/rectangular_aperture.png',
         device = torch.device('cpu'),
         output_directory = 'output'
        ):
    target_points = get_target_plane_points(
                                            resolution = resolution,
                                            resolution_factor = resolution_factor,
                                            z = distance,
                                            pixel_pitch = pixel_pitch,
                                            device = device
                                           )
    aperture_pattern = odak.learn.tools.load_image(aperture_pattern_filename, normalizeby = 255., torch_style = True).to(device)[0]
    aperture_points, aperture_field = get_aperture_points(
                                                          aperture_pattern,
                                                          dimensions = [pixel_pitch, pixel_pitch],
                                                          device = device
                                                         )
    h = torch.zeros(
                    resolution[0] * resolution_factor,
                    resolution[1] * resolution_factor,
                    dtype = torch.complex64,
                    device = device
                   )
    for i in range(nx):
        for j in range(ny):
            shift = torch.tensor(
                                 [[
                                   pixel_pitch / nx * i - pixel_pitch / 2.,
                                   pixel_pitch / ny * j - pixel_pitch / 2.,
                                   0.,
                                 ]],
                                 device = device
                                )
            h += odak.learn.wave.get_point_wise_impulse_response_fresnel_kernel(
                                                                                aperture_points = aperture_points,
                                                                                aperture_field = aperture_field,
                                                                                target_points = target_points + shift,
                                                                                resolution = resolution,
                                                                                resolution_factor = resolution_factor,
                                                                                wavelength = wavelength,
                                                                                distance = distance,
                                                                                randomization = randomization,
                                                                                device = device
                                                                               )
    h = h / nx / ny
    save_psfs(
              h,
              directory = '{}/aperture_psfs/'.format(output_directory)
             )
    assert True == True


def save_psfs(kernel, directory, wavelength_id = 0, distance_id = 0, pixel_pitch_id = 0):
    odak.tools.check_directory(directory)
    kernel_amplitude = odak.learn.wave.calculate_amplitude(kernel)
    kernel_intensity = kernel_amplitude ** 2
    kernel_amplitude = kernel_amplitude / kernel_amplitude.max()
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


if __name__ == '__main__':
    sys.exit(main())
