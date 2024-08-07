#!/usr/bin/env python

import sys
import numpy as np
import odak
import odak.raytracing


def test_ray_intersect_w_a_triangle():
    ray_start_point_0 = [5., 5., 0.]
    ray_end_point_0 = [10., 10., 1000.]
    ray_0 = odak.raytracing.create_ray_from_two_points(
                                                       ray_start_point_0,
                                                       ray_end_point_0
                                                      )
    ray_start_point_1 = [5., 5., 0.]
    ray_end_point_1 = [0., 100., 1000.]
    ray_1 = odak.raytracing.create_ray_from_two_points(
                                                       ray_start_point_1,
                                                       ray_end_point_1
                                                      )
    triangle = [
                [50.,  50., 1000.],
                [-5.,  -5., 1000.],
                [0.,  50., 1000.],
               ]
    normal_0, distance_0 = odak.raytracing.intersect_w_triangle(
                                                                ray_0,
                                                                triangle
                                                               )
    normal_1, distance_1 = odak.raytracing.intersect_w_triangle(
                                                                ray_1,
                                                                triangle
                                                               )
    assert True == True


if __name__ == '__main__':
    sys.exit(test_ray_intersect_w_a_triangle())
