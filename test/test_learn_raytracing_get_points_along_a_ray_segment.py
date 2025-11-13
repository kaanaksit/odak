import sys
import odak
import torch


def test(directory = 'test_output'):
    odak.tools.check_directory(directory)
    starting_point = torch.tensor([[0., 0., 0.]])
    end_point = torch.tensor([[5., 5., 5.]])
    distances = torch.linspace(0., 10., 10)


    ray = odak.learn.raytracing.create_ray_from_two_points(
                                                           starting_point,
                                                           end_point
                                                          )
    points = odak.learn.raytracing.get_points_along_a_ray_segment(
                                                                  ray = ray,
                                                                  distances = distances
                                                                 )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
