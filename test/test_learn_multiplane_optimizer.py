import sys


def test():
    import torch
    from odak.learn.wave import multiplane_hologram_optimizer, multiplane_loss
    resolution = [1080, 1920]
    target = torch.zeros(resolution[0], resolution[1])
    target[500::600, :] = 1
    depth = target
    loss_function = multiplane_loss(
                                    target_image = target.unsqueeze(0),
                                    target_depth = depth,
                                    target_blur_size = 20,
                                    number_of_planes = 8,
                                    multiplier = 1.0,
                                    blur_ratio = 3,
                                    weights = [1.0, 1.0, 1.0, 0.0],
                                    scheme = "defocus",
                                    reduction = 'mean'
                                   )
    targets, focus_target, depth = loss_function.get_targets()
    optimizer = multiplane_hologram_optimizer(
                                              wavelength = 0.000000515,
                                              image_location = 0.0,
                                              image_spacing = 0.001,
                                              slm_pixel_pitch = 0.000008,
                                              slm_resolution = resolution,
                                              targets = targets,
                                              propagation_type = "Bandlimited Angular Spectrum",
                                              number_of_iterations = 2,
                                              learning_rate = 0.04,
                                              number_of_planes = 2,
                                              mask_limits = [0.0, 1.0, 0.0, 1.0], 
                                              zero_mode_distance = 0.3,
                                              loss_function = loss_function,
                                              optimize_amplitude = True
                                             )
    phase, amplitude, reconstructions = optimizer.optimize()
    assert True == True


if __name__ == '__main__':
    sys.exit(test())