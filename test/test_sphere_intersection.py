#!/usr/bin/env python

import sys

def test_sphere_intersection():
    import odak.raytracing as raytracer
    ray             = raytracer.create_ray_from_two_points([-2.,0.,0.],[10.,0.,0.])
    sphere          = raytracer.define_sphere([0.,0.,0.],10)
    distance,normal = raytracer.intersect_w_sphere(ray,sphere)
    assert True==True

def test_multiple_rays_w_sphere():
    import odak.raytracing as raytracer
    import odak.tools as tools
    start_points      = tools.grid_sample(
                                          no=[10,10],
                                          size=[100.,100.],
                                          center=[0.,0.,0.],
                                          angles=[0.,0.,0.]
                                         )
    end_point         = [0.,0.,100.]
    rays              = raytracer.create_ray_from_two_points(
                                                             start_points,
                                                             end_point
                                                            )
    sphere            = raytracer.define_sphere([0.,0.,100.],20)
    distances,normals = raytracer.intersect_w_sphere(rays,sphere)
    assert True==True

def test_all():
    test_sphere_intersection()
    test_multiple_rays_w_sphere()

if __name__ == '__main__':
    sys.exit(test_all())
