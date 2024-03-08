#!/usr/bin/env python

import sys
import odak
import torch

def test():
    rays_from_grid_w_luminous_angle = odak.raytracing.create_ray_from_grid_w_luminous_angle(
        center = torch.tensor([0, 0, 0]),
        size = [2, 2],
        no = [2, 2],
        tilt = torch.tensor([15, 0, 0]),
        num_ray_per_light = 10,
        angle_limit = 15,
    )

    distances = torch.ones(rays_from_grid_w_luminous_angle.shape[0]) * 12.5
    propagated_rays = odak.learn.raytracing.propagate_ray(
                                                          rays_from_grid_w_luminous_angle,
                                                          distances
                                                         )


    visualize = True
    if visualize:
        ray_diagram = odak.visualize.plotly.rayshow(line_width = 3., marker_size = 3.)
        ray_diagram.add_point(rays_from_grid_w_luminous_angle[:, 0], color = 'red')
        ray_diagram.add_point(propagated_rays[:, 0], color = 'blue')

        ray_diagram.add_line(rays_from_grid_w_luminous_angle[:, 0], propagated_rays[:, 0])

        html = ray_diagram.save_offline()
        markdown_file = open('ray.txt', 'w')
        markdown_file.write(html)
        markdown_file.close()
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
