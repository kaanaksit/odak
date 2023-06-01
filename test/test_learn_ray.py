#!/usr/bin/env python

import sys
import odak
import torch # (1)


def test():
    starting_point = torch.tensor([5., 5., 0.]) # (2)
    end_points, _, _, _ = odak.learn.tools.grid_sample(
                                                       no = [2, 2], 
                                                       size = [20., 20.], 
                                                       center = [0., 0., 10.]
                                                      ) # (3)
    rays_from_points = odak.learn.raytracing.create_ray_from_two_points(
                                                                        starting_point,
                                                                        end_points
                                                                       ) # (4)


    starting_points, _, _, _ = odak.learn.tools.grid_sample(
                                                            no = [3, 3], 
                                                            size = [100., 100.], 
                                                            center = [0., 0., 10.],
                                                           )
    angles = torch.randn_like(starting_points) * 180. # (5)
    rays_from_angles = odak.learn.raytracing.create_ray(
                                                        starting_points,
                                                        angles
                                                       ) # (6)


    distances = torch.ones(rays_from_points.shape[0]) * 12.5
    propagated_rays = odak.learn.raytracing.propagate_a_ray(
                                                            rays_from_points,
                                                            distances
                                                           ) # (7)
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
