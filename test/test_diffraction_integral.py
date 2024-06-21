import sys
import odak # (1)
import torch
from tqdm import tqdm


def main(): # (2)
    length = [7e-6, 7e-6] # (3)
    for fresnel_id, fresnel_number in enumerate(range(99)): # (4)
        fresnel_number += 1
        propagate(
                  fresnel_number = fresnel_number,
                  length = [length[0] + 1. / fresnel_number * 8e-6, length[1] + 1. / fresnel_number * 8e-6]
                 )


def propagate(
              wavelength = 532e-9, # (6)
              pixel_pitch = 3.74e-6, # (7)
              length = [15e-6, 15e-6],
              image_samples = [2, 2], # Replace it with 1000 by 1000 (8)
              aperture_samples = [2, 2], # Replace it with 1000 by 1000 (9)
              device = torch.device('cpu'),
              output_directory = 'test_output', 
              fresnel_number = 4,
              save_flag = False
             ): # (5)
    distance = pixel_pitch ** 2 / wavelength / fresnel_number
    distance = torch.as_tensor(distance, device = device)
    k = odak.learn.wave.wavenumber(wavelength)
    x = torch.linspace(- length[0] / 2, length[0] / 2, image_samples[0], device = device)
    y = torch.linspace(- length[1] / 2, length[1] / 2, image_samples[1], device = device)
    X, Y = torch.meshgrid(x, y, indexing = 'ij') # (10)
    wxs = torch.linspace(- pixel_pitch / 2., pixel_pitch / 2., aperture_samples[0], device = device)
    wys = torch.linspace(- pixel_pitch / 2., pixel_pitch / 2., aperture_samples[1], device = device) # (11)
    h  = torch.zeros(image_samples[0], image_samples[1], dtype = torch.complex64, device = device)
    for wx in tqdm(wxs):
        for wy in wys:
            h += huygens_fresnel_principle(wx, wy, X, Y, distance, k, wavelength) # (12)
    h = h * pixel_pitch ** 2 / aperture_samples[0] / aperture_samples[1] # (13) 

    if save_flag:
        save_results(h, output_directory, fresnel_number, length, pixel_pitch, distance, image_samples, device) # (14)
    return True


def huygens_fresnel_principle(x, y, X, Y, z, k, wavelength): # (12)
    r = torch.sqrt((X - x) ** 2 + (Y - y) ** 2 + z ** 2)
    h = torch.exp(1j * k * r) * z / r ** 2 * (1. / (2 * odak.pi * r) + 1. / (1j * wavelength))
    return h


def save_results(h, output_directory, fresnel_number, length, pixel_pitch, distance, image_samples, device):
    from matplotlib import pyplot as plt
    odak.tools.check_directory(output_directory)
    output_intensity = odak.learn.wave.calculate_amplitude(h) ** 2
    odak.learn.tools.save_image(
                                '{}/diffraction_output_intensity_fresnel_number_{:02d}.png'.format(output_directory, int(fresnel_number)),
                                output_intensity,
                                cmin = 0.,
                                cmax = output_intensity.max()
                               )
    cross_section_1d = output_intensity[output_intensity.shape[0] // 2]
    lengths = torch.linspace(- length[0] * 10 ** 6 / 2., length[0] * 10 ** 6 / 2., image_samples[0], device = device)
    plt.figure()
    plt.plot(lengths.detach().cpu().numpy(), cross_section_1d.detach().cpu().numpy())
    plt.xlabel('length (um)')
    plt.figtext(
                0.14,
                0.9, 
                r'Fresnel Number: {:02d}, Pixel pitch: {:.2f} um, Distance: {:.2f} um'.format(fresnel_number, pixel_pitch * 10 ** 6, distance * 10 ** 6),
                fontsize = 11
               )
    plt.savefig('{}/diffraction_1d_output_intensity_fresnel_number_{:02d}.png'.format(output_directory, int(fresnel_number)))
    plt.cla()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    sys.exit(main())
