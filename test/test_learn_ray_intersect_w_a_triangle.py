#!/usr/bin/env python

import sys
import odak
import torch
import odak.visualize.plotly


def test():
    starting_points, _, _, _ = odak.learn.tools.grid_sample(
                                                            no = [5, 5],
                                                            size = [10., 10.],
                                                            center = [0., 0., 0.]
                                                           )
    end_points, _, _, _ = odak.learn.tools.grid_sample(
                                                       no = [5, 5],
                                                       size = [5., 5.],
                                                       center = [0., 0., 10.]
                                                      )
    rays = odak.learn.raytracing.create_ray_from_two_points(
                                                            starting_points,
                                                            end_points
                                                           )
    triangle = torch.tensor(
                            [[
                              [-5., -5., 10.],
                              [ 5., -5., 10.],
                              [ 0.,  5., 10.]
                            ]]
                           )
    normals, distance, check = odak.learn.raytracing.intersect_w_triangle(
                                                                          rays,
                                                                          triangle
                                                                         )
    """
    ray_diagram = odak.visualize.plotly.rayshow(line_width = 3., marker_size = 3.)
    ray_diagram.add_triangle(triangle, color = 'black')
    ray_diagram.add_point(rays[:, 0], color = 'blue')
    ray_diagram.add_line(rays[:, 0], normals[:, 0], color = 'blue')
    colors = []
    for color_id in range(check.shape[0]):
        if check[color_id] == True:
            colors.append('green')
        elif check[color_id] == False:
            colors.append('red')
    ray_diagram.add_point(normals[:, 0], color = colors)
    ray_diagram.show()
    """
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
