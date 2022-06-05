#!/usr/bin/env python

import sys


def test_plano_convex():
    import odak
    import odak.raytracing as raytracer
    import odak.tools as tools
    import odak.catalog as catalog
    end_points = tools.grid_sample(
        no=[5, 5],
        size=[2.0, 2.0],
        center=[0., 0., 0.],
        angles=[0., 0., 0.]
    )
    start_point = [0., 0., -5.]
    rays = raytracer.create_ray_from_two_points(
        start_point,
        end_points
    )
    lens = catalog.plano_convex_lens()
    normals, distances = lens.intersect(rays)
    assert True == True


if __name__ == '__main__':
    sys.exit(test_plano_convex())
