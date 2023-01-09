#!/usr/bin/env python

import sys
import odak
import odak.learn.raytracing
import torch


def test():
    # Input points to create a ray.
    ray_start_point_0 = torch.tensor([5., 5., 0.])
    ray_end_point_0 = torch.tensor([10., 10., 1000.])
    # Create from two points.
    ray_0 = odak.learn.raytracing.create_ray_from_two_points(
        ray_start_point_0,
        ray_end_point_0
    )
    # Input points to create a ray.
    ray_start_point_1 = torch.tensor([5., 5., 0.])
    ray_end_point_1 = torch.tensor([0., 100., 1000.])
    # Create from two points.
    ray_1 = odak.learn.raytracing.create_ray_from_two_points(
        ray_start_point_1,
        ray_end_point_1
    )
    # Intersection with a triangle.
    triangle = torch.tensor([
        [50.,  50., 1000.],
        [-5.,  -5., 1000.],
        [0.,  50., 1000.],
    ])
    normal_0, distance_0 = odak.learn.raytracing.intersect_w_triangle(
        ray_0,
        triangle
    )
    normal_1, distance_1 = odak.learn.raytracing.intersect_w_triangle(
        ray_1,
        triangle
    )
    print(normal_0)
    print(normal_1)
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
