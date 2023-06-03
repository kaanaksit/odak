#!/usr/bin/env python


import sys
import odak
import torch


def test():
    starting_points, _, _, _ = odak.learn.tools.grid_sample(
                                                            no = [5, 5],
                                                            size = [20., 20.],
                                                            center = [0., 0., 0.]
                                                           )
    end_points, _, _, _ = odak.learn.tools.grid_sample(
                                                       no = [5, 5],
                                                       size = [6., 6.],
                                                       center = [0., 0., 10.]
                                                      )
    rays = odak.learn.raytracing.create_ray_from_two_points(
                                                            starting_points,
                                                            end_points
                                                           )
    triangle = torch.tensor([[
                              [-5., -5., 10.],
                              [ 5., -5., 10.],
                              [ 0.,  5., 10.]
                            ]])
    normals, distance, intersecting_rays, intersecting_normals, check = odak.learn.raytracing.intersect_w_triangle(
                                                                                    rays,
                                                                                    triangle
                                                                                   ) 
    n_air = 1.0; n_glass = 1.51
    refracted_rays = odak.learn.raytracing.refract(intersecting_rays, intersecting_normals, n_air, n_glass)
    propagated_rays = odak.learn.raytracing.propagate_ray(
                                                          refracted_rays, 
                                                          torch.ones(refracted_rays.shape[0]) * 10.
                                                         )
    visualize = True
    if visualize:
        ray_diagram = odak.visualize.plotly.rayshow(line_width = 3., marker_size = 3.) # (1)
        ray_diagram.add_triangle(triangle, color = 'black')
        ray_diagram.add_point(rays[:, 0], color = 'blue')
        ray_diagram.add_line(rays[:, 0], normals[:, 0], color = 'blue')
        ray_diagram.add_line(refracted_rays[:, 0], propagated_rays[:, 0], color = 'blue')
        colors = []
        for color_id in range(check.shape[1]):
            if check[0, color_id] == True:
                colors.append('green')
            elif check[0, color_id] == False:
                colors.append('red')
        ray_diagram.add_point(normals[:, 0], color = colors)
        html = ray_diagram.save_offline()
        print(html)
    assert True == True
   

if __name__ == '__main__':
    sys.exit(test())
