import sys
import odak.raytracing as raytracer
from odak.tools.sample import grid_sample, batch_of_rays


def test():
    sample_entry_points = grid_sample(
                                      no = [4, 4],
                                      size = [100., 100.],
                                      center = [0., 0., 0.],
                                      angles = [0., 0., 0.]
                                     )
    sample_exit_points = grid_sample(
                                     no = [4, 4],
                                     size = [100., 100.],
                                     center = [0., 0., 100.],
                                     angles = [0., 0., 0.]
                                    )
    rays = raytracer.create_ray_from_two_points(
                                                sample_entry_points,
                                                sample_exit_points
                                               )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
