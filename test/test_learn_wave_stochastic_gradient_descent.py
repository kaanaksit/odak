import sys
import odak
import torch


def test(output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    device = torch.device('cpu') # (1)
    target = odak.learn.tools.load_image('./test/data/usaf1951.png', normalizeby = 255., torch_style = True)[1] # (4)
    hologram, reconstruction = odak.learn.wave.stochastic_gradient_descent(
                                                                           target,
                                                                           wavelength = 532e-9,
                                                                           distance = 20e-2,
                                                                           pixel_pitch = 8e-6,
                                                                           propagation_type = 'Bandlimited Angular Spectrum',
                                                                           n_iteration = 50,
                                                                           learning_rate = 0.1
                                                                          ) # (2)
    odak.learn.tools.save_image(
                                '{}/phase.png'.format(output_directory), 
                                odak.learn.wave.calculate_phase(hologram) % (2 * odak.pi), 
                                cmin = 0., 
                                cmax = 2 * odak.pi
                               ) # (3)
    odak.learn.tools.save_image('{}/sgd_target.png'.format(output_directory), target, cmin = 0., cmax = 1.)
    odak.learn.tools.save_image(
                                '{}/sgd_reconstruction.png'.format(output_directory), 
                                odak.learn.wave.calculate_amplitude(reconstruction) ** 2, 
                                cmin = 0., 
                                cmax = 1.
                               )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
