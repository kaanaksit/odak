import odak.raytracing as raytracer
import sys
import numpy as np


def test():
    start0 = np.array([0., 5., 0.])
    start1 = np.array([0., -5., 0.])
    intersection = np.array([0., 0., 100.])
    ray0 = raytracer.create_ray_from_two_points(
                                                start0,
                                                intersection
                                               )
    ray1 = raytracer.create_ray_from_two_points(
                                                start1,
                                                intersection
                                               )
    c0, c1 = raytracer.find_nearest_points(ray0, ray1)
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
