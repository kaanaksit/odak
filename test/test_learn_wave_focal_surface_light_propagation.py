import sys
import os
import odak
import torch
import requests


def test(output_directory = 'test_output'):
    number_of_planes = 6
    location_offset = 0.
    volume_depth = 5e-3
    device = torch.device('cpu')

    # Download the weight and key mapping files from GitHub

    weight_url = 'https://raw.githubusercontent.com/complight/focal_surface_holographic_light_transport/main/weight/model_0mm.pt'
    key_mapping_url = 'https://raw.githubusercontent.com/complight/focal_surface_holographic_light_transport/main/weight/key_mappings.json'
    weight_filename = os.path.join(output_directory, 'model_0mm.pt')
    key_mapping_filename = os.path.join(output_directory, 'key_mappings.json')
    download_file(weight_url, weight_filename)
    download_file(key_mapping_url, key_mapping_filename)

    # Preparing focal surface
    focal_surface_filename = os.path.join(output_directory, 'sample_0343_focal_surface.png')
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
    hologram_phases_filename = os.path.join(output_directory, 'sample_0343_hologram.png')
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
    print("Reconstruction complete.")
    return True


# Function to download a file from GitHub
def download_file(url, filename):
    try:
        print(f"Starting download: {url}")
        response = requests.get(url, stream = True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(filename), exist_ok = True)
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size = 8192):
                file.write(chunk)
        print(f"Downloaded: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}. Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(test())
