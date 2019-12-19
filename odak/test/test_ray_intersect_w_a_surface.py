#!/usr/bin/env python

import sys
import numpy as np
import odak
import odak.raytracing

def test_ray_intersect_w_a_surface():
    # Input points to create a ray.
    ray_start_point    = [5.,5.,0.]
    ray_end_point      = [0.,0.,1000.]
    print('Input starting point: %s' % ray_start_point)
    print('Input end point: %s' % ray_end_point)
    # Create from two points.
    ray                = odak.raytracing.create_ray_from_two_points(
                                                                    ray_start_point,
                                                                    ray_end_point
                                                                   )
    print('Starting point of the created ray: %s' % ray[0])
    print('Angles of the created ray: %s' % ray[1])
    # Intersection with the surface.
    surface_points     = [
                          [10.,10.,1000.],
                          [ 0.,10.,1000.],
                          [ 0., 0.,1000.]
                         ] # <-- Three points on a surface.
    print('Points to define surface: %s %s %s' % (surface_points[0],surface_points[1],surface_points[2]))
    normal,distance    = odak.raytracing.find_intersection_w_surface(
                                                                     ray,
                                                                     surface_points
                                                                    )
    print('Intersection point: %s' % normal[0])
    print('Surface normal angles: %s' % normal[1])
    assert True==True

if __name__ == '__main__':
    sys.exit(test_ray_intersect_w_a_surface())
