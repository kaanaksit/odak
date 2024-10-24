import sys
import os
import odak
import torch


def test(output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    number_of_planes = 6
    location_offset = 0.
    volume_depth = 5e-3
    device = torch.device('cpu')

    weight_filename = 'test/data/focal_surface_sample_model.pt'
    key_mapping_filename = 'test/data/key_mappings.json'

    # Preparing focal surface
    focal_surface_filename = 'test/data/sample_0343_focal_surface.png'
    focal_surface = odak.learn.tools.load_image(
                                                focal_surface_filename,
                                                normalizeby = 255.,
                                                torch_style = True
                                               ).to(device)
    distances = torch.linspace(-volume_depth / 2., volume_depth / 2., number_of_planes) + location_offset
    y = (distances - torch.min(distances))
    distances = (y / torch.max(y))
    focal_surface = focal_surface * (number_of_planes - 1)
    focal_surface = torch.round(focal_surface, decimals = 0)
    for i in range(number_of_planes):
        focal_surface = torch.where(focal_surface == i, distances[i], focal_surface)
    focal_surface = focal_surface.unsqueeze(0).unsqueeze(0)

    # Preparing hologram
    hologram_phases_filename = 'test/data/sample_0343_hologram.png'
    hologram_phases = odak.learn.tools.load_image(
                                                  hologram_phases_filename,
                                                  normalizeby = 255.,
                                                  torch_style = True
                                                 ).to(device)
    hologram_phases = hologram_phases.unsqueeze(0)

    # Load the focal surface light propagation model
    focal_surface_light_propagation_model = odak.learn.wave.focal_surface_light_propagation(device = device)
    focal_surface_light_propagation_model.load_weights(
                                                       weight_filename = weight_filename,
                                                       key_mapping_filename = key_mapping_filename
                                                      )

    # Perform the focal surface light propagation model
    result = focal_surface_light_propagation_model(focal_surface, hologram_phases)

    odak.learn.tools.save_image(
                                '{}/reconstruction_image.png'.format(output_directory),
                                result,
                                cmin = 0.,
                                cmax = 1.
                               )
    return True


if __name__ == '__main__':
    sys.exit(test())
