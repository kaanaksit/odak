#!/usr/bin/env python

import sys
import odak
import torch


def test():
    starting_point = torch.tensor([5., 5., 0.]).unsqueeze(0)
    end_points, _, _, _ = odak.learn.tools.grid_sample(
                                                       no = [3, 3], 
                                                       size = [20., 20.], 
                                                       center = [0., 0., 10.]
                                                      )
    rays_from_points = odak.learn.raytracing.create_ray_from_two_points(
                                                                        starting_point,
                                                                        end_points
                                                                       )


    starting_points, _, _, _ = odak.learn.tools.grid_sample(
                                                            no = [4, 4], 
                                                            size = [100., 100.], 
                                                            center = [0., 0., 10.],
                                                           )
    angles = torch.randn_like(starting_points)
    rays_from_angles = odak.learn.raytracing.create_ray(
                                                        starting_points,
                                                        angles
                                                       )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
