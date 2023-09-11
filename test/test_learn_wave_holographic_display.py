import sys
import odak
import torch


def test():
    device = torch.device('cpu')
    display = odak.learn.wave.holographic_display(
                                                  wavelengths = [639e-9, 515e-9, 473e-9],
                                                  pixel_pitch = 3.74e-6,
                                                  resolution = [2400, 4094],
                                                  volume_depth = 0.01,
                                                  number_of_depth_layers = 2,
                                                  image_location_offset = 5e-3,
                                                  pinhole_size = 1500,
                                                  propagation_type = 'Bandlimited Angular Spectrum',
                                                  device = device
                                                 )
    hologram_phases = odak.learn.tools.load_image(
                                                  './test/sample_hologram.png', 
                                                  normalizeby = 255., 
                                                  torch_style = True
                                                 ) * odak.pi * 2.
    laser_powers = torch.tensor([
                                 [1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.],
                                ], 
                                device = device
                               )
    reconstructions = display.reconstruct(hologram_phases = hologram_phases.to(device), laser_powers = laser_powers)
    combined_frame = torch.sum(reconstructions, dim = 0)
    for depth_id in range(reconstructions.shape[1]):
        odak.learn.tools.save_image(
                                    'reconstructions_{:04d}.png'.format( depth_id),
                                    combined_frame[depth_id],
                                    cmin = 0., 
                                    cmax = 1.
                                   )                                    
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
