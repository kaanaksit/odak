import odak.raytracing as raytracer
import sys
import numpy as np


def detector_to_light_source(detector_location, triangle, light_source):
    center_of_triangle = raytracer.center_of_triangle(triangle)
    ray_detector_triangle = raytracer.create_ray_from_two_points(
        detector_location,
        center_of_triangle
    )
    normal_triangle, d_det_tri = raytracer.intersect_w_triangle(
        ray_detector_triangle,
        triangle
    )
    if d_det_tri == 0:
        return 0
    ray_reflection = raytracer.reflect(
        ray_detector_triangle,
        normal_triangle
    )
    normal_source, d_tri_sou = raytracer.intersect_w_circle(
        ray_reflection,
        light_source
    )
    if d_tri_sou == 0:
        return 0
    opl = d_det_tri + d_tri_sou
    return opl


def test():
    detector_location = [2., 0., 0.]
    triangle = np.array(
        [
            [10., 10., 10.],
            [0., 10., 10.],
            [0.,  0., 10.]
        ]
    )
    circle_center = [0., 0., 0.]
    circle_angles = [0., 0., 0.]
    circle_radius = 15.
    circle = raytracer.define_circle(
        angles=circle_angles,
        center=circle_center,
        radius=circle_radius
    )

    opl = detector_to_light_source(
        detector_location,
        triangle,
        circle
    )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
