import torch
import sys

from os.path import join
from odak.learn.perception.util import slice_rgbd_targets
from odak.learn.tools import load_image, save_image
from odak.tools import check_directory


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
    depth = torch.mean(depth, dim = 0) # Ensure the depthmap has the shape of [w x h]

    depth_plane_positions = torch.linspace(0, 1, steps=5).to(device)
    targets, masks = slice_rgbd_targets(target, depth, depth_plane_positions)
    depth_slices_sum = torch.zeros_like(target)
    for idx, target in enumerate(targets):
        depth_slices_sum += masks[idx]
        save_image(
                   join(output_directory, f"target_{idx}.png"),
                   target,
                   cmin = target.min(),
                   cmax = target.max()
                  )
        save_image(
                   join(output_directory, f"depth_{idx}.png"),
                   masks[idx],
                   cmin = masks[idx].min(),
                   cmax = masks[idx].max()
                  )
    print(depth_slices_sum.mean().item())
    assert depth_slices_sum.mean().item() == 1. # The mean of the depth slices sum shoud be 1
    
    
if __name__ == '__main__':
    sys.exit(test())