#!/usr/bin/env python


import sys
import odak
import torch


def test():
    final_surface = torch.tensor([[
                                   [-5., -5., 0.],
                                   [ 5., -5., 0.],
                                   [ 0.,  5., 0.]
                                 ]])
    final_target = torch.tensor([[20., 20., 0.]])
    triangle = torch.tensor([[
                              [-5., -5., 10.],
                              [ 5., -5., 10.],
                              [ 0.,  5., 10.]
                            ]])
    starting_points, _, _, _ = odak.learn.tools.grid_sample(
                                                            no = [5, 5],
                                                            size = [10., 10.],
                                                            center = [0., 0., 0.]
                                                           )
    end_point = odak.learn.raytracing.center_of_triangle(triangle)
    rays = odak.learn.raytracing.create_ray_from_two_points(
                                                            starting_points,
                                                            end_point
                                                           )
    angles = torch.zeros(1, 3, requires_grad = True)
    learning_rate = 2e-3
    optimizer = torch.optim.Adam([angles], lr = learning_rate)
    loss_function = torch.nn.MSELoss()
    number_of_steps = 1
    for step in range(number_of_steps):
        rotated_triangle = odak.learn.tools.rotate_points(
                                                          triangle.squeeze(0), 
                                                          angles = angles, 
                                                          origin = end_point
                                                         )
        _, _, intersecting_rays, intersecting_normals, check = odak.learn.raytracing.intersect_w_triangle(
                                                                                                          rays,
                                                                                                          triangle
                                                                                                         )
        reflected_rays = odak.learn.raytracing.reflect(intersecting_rays, intersecting_normals)
        intersection_normals, _ = odak.learn.raytracing.intersect_w_surface(reflected_rays, final_surface)
        intersection_points = intersection_normals[:, 0]
    assert True == True
      

if __name__ == '__main__':
    sys.exit(test())
