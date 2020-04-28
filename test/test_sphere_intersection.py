#!/usr/bin/env python

import sys

def test_sphere_intersection():
    import odak.raytracing as raytracer
    ray             = raytracer.create_ray_from_two_points([-2.,0.,0.],[10.,0.,0.])
    sphere          = raytracer.define_sphere([0.,0.,0.],10)
    distance,normal = raytracer.intersect_w_sphere(ray,sphere)
    assert True==True

if __name__ == '__main__':
    sys.exit(test_sphere_intersection())
