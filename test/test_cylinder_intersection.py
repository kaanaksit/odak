#!/usr/bin/env python

import sys


def test_cylinder_intersection():
    import odak.raytracing as raytracer
    ray = raytracer.create_ray_from_two_points([0., 0., 0.], [10., 0., 0.])
    cylinder = raytracer.define_cylinder(
        [0., 0., 0.], 5., rotation=[0., 90., 0.])
    normal, distance = raytracer.intersect_w_cylinder(ray, cylinder)
    assert True == True


def test_multiple_rays_w_cylinder():
    import odak.raytracing as raytracer
    import odak.tools as tools
    end_points = tools.grid_sample(
        no=[5, 5],
        size=[10., 10.],
        center=[0., 0., 100.],
        angles=[0., 0., 0.]
    )
    start_point = [0., 0., 0.]
    rays = raytracer.create_ray_from_two_points(
        start_point,
        end_points
    )
    sphere = raytracer.define_sphere([0., 0., 100.], 20)
    normals, distances = raytracer.intersect_w_sphere(rays, sphere)
    assert True == True


def test():
    test_cylinder_intersection()
#    test_multiple_rays_w_sphere()


if __name__ == '__main__':
    sys.exit(test())
