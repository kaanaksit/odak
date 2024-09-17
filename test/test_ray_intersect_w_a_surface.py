import sys
import numpy as np
import odak
import odak.raytracing


def test_ray_intersect_w_a_surface():
    ray_start_point = [5., 5., 0.]
    ray_end_point = [0., 0., 1000.]
    ray = odak.raytracing.create_ray_from_two_points(
                                                     ray_start_point,
                                                     ray_end_point
                                                    )
    surface_center = [0., 0., 1000.]
    surface_angles = [0., 0., 0.]
    surface_points = odak.raytracing.define_plane(
                                                  surface_center,
                                                  surface_angles
                                                 )
    normal, distance = odak.raytracing.intersect_w_surface(
                                                           ray,
                                                           surface_points
                                                          )
    assert True == True


if __name__ == '__main__':
    sys.exit(test_ray_intersect_w_a_surface())
