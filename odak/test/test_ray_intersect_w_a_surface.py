#!/usr/bin/env python

import sys
import numpy as np
import odak

def test_ray_intersect_w_a_surface():
    # Create a ray.
    ray_starting_point = [0.,0.,0.]
    ray_angles         = [45.,0.,0.]
    ray                = odak.raytracing.create_ray(
                                                    ray_starting_point,
                                                    ray_angles
                                                   )
    # Intersect the created ray with a surface.
    surface_points     = np.array(
                                  [
                                   [10,10,1000],
                                   [-10,10,1000],
                                   [0,10,1000]
                                  ]
                                  ) # <-- Three points on a surface.                           
    normal,distance    = odak.raytracing.find_intersection_w_surface(
                                                                     ray,
                                                                     surface_points
                                                                    )
    assert True==True

if __name__ == '__main__':
    sys.exit(test_ray_intersect_w_a_surface())
