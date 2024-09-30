import torch
import sys

from os.path import join
from odak.tools import check_directory
from odak.learn.tools import load_image, save_image
from odak.learn.perception.util import slice_rgbd_targets
from odak.learn.wave import incoherent_focal_stack_rgbd


def test(device = torch.device('cpu'), output_directory = 'test_output'):
    check_directory(output_directory)
    
    target = load_image(
                        "test/data/sample_rgb.png",
                        normalizeby = 256,
                        torch_style = True
                       ).to(device)

    depth = load_image(
                       "test/data/sample_depthmap.png",
                       normalizeby = 256,
                       torch_style = True
                      ).to(device)
    depth = torch.mean(depth, dim = 0).to(device) # Ensure the depthmap has the shape of [w x h]
    depth_plane_positions = torch.linspace(0, 1, steps=5).to(device)
    
    wavelengths = torch.tensor([639e-9, 515e-9, 473e-9], dtype = torch.float32).to(device)
    pixel_pitch = 3.74e-6
    
    targets, masks = slice_rgbd_targets(target, depth, depth_plane_positions)
    distances = depth_plane_positions * 0.0005 # Multiply with some multiplier to control the blurriness
    
    focal_stack = incoherent_focal_stack_rgbd(
                                              targets = targets,
                                              masks = masks,
                                              distances = distances, 
                                              dx = pixel_pitch, 
                                              wavelengths = wavelengths,
                                             )

    for idx, focal_image in enumerate(focal_stack):
        save_image(
                   join(output_directory, f"focal_target_{idx}.png"),
                   focal_image,
                   cmin = focal_image.min(),
                   cmax = focal_image.max()
                  )
    assert True == True
    
    
if __name__ == '__main__':
    sys.exit(test())