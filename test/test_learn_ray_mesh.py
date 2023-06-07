#!/usr/bin/env python


import sys
import odak
import torch
from tqdm import tqdm


def test():
    device = torch.device('cpu')
    final_target = torch.tensor([-2., -2., 10.], device = device)
    final_surface = odak.learn.raytracing.define_plane(point = final_target)
    mesh = odak.learn.raytracing.planar_mesh(size = [1.2, 1.2], number_of_meshes = [5, 5], device = device)
    start_points, _, _, _ = odak.learn.tools.grid_sample(
                                                         no = [20, 20],
                                                         size = [1., 1.],
                                                         center = [2., 2., 10.]
                                                        )
    end_points, _, _, _ = odak.learn.tools.grid_sample(
                                                       no = [20, 20],
                                                       size = [1., 1.],
                                                       center = [0., 0., 0.]
                                                      )
    start_points = start_points.to(device)
    end_points = end_points.to(device)
    loss_function = torch.nn.MSELoss()
    learning_rate = 2e-3
    optimizer = torch.optim.AdamW([mesh.heights], lr = learning_rate)
    rays = odak.learn.raytracing.create_ray_from_two_points(start_points, end_points)
    number_of_steps = 100
    t = tqdm(range(number_of_steps), leave = False, dynamic_ncols = True)
    for step in t:
        optimizer.zero_grad()
        triangles = mesh.get_triangles()
        reflected_rays, reflected_normals = mesh.mirror(rays)
        final_normals, _ = odak.learn.raytracing.intersect_w_surface(reflected_rays, final_surface)
        final_points = final_normals[:, 0]
        target = final_target.repeat(final_points.shape[0], 1)
        if step == 0:
            start_triangles = triangles.detach().clone()
            start_reflected_rays = reflected_rays.detach().clone()
            start_final_normals = final_normals.detach().clone()
        loss = loss_function(final_points, target)
        loss.backward(retain_graph = True)
        optimizer.step() 
        description = 'Loss: {}'.format(loss.item())
        t.set_description(description)
    print(description)
            

    visualize = False
    if visualize:
        ray_diagram = odak.visualize.plotly.rayshow(
                                                    rows = 2,
                                                    columns = 2,
                                                    line_width = 3.,
                                                    marker_size = 1.,
                                                    subplot_titles = ['Before optimization', 'After optimization']
                                                   ) 
        for triangle_id in range(triangles.shape[0]):
            ray_diagram.add_triangle(start_triangles[triangle_id], row = 1, column = 1, color = 'black')
            ray_diagram.add_triangle(triangles[triangle_id], row = 1, column = 2, color = 'black')
            ray_diagram.add_point(start_reflected_rays[:, 0], row = 1, column = 1, color = 'red')
            ray_diagram.add_point(reflected_rays[:, 0], row = 1, column = 2, color = 'red')
            ray_diagram.add_point(start_final_normals[:, 0], row = 2, column = 1, color = 'blue')
            ray_diagram.add_point(final_target, row = 2, column = 1, color = 'green')
            ray_diagram.add_point(final_normals[:, 0], row = 2, column = 2, color = 'blue')
            ray_diagram.add_point(final_target, row = 2, column = 2, color = 'green')
        html = ray_diagram.save_offline()
        print(html)
    assert True == True
      

if __name__ == '__main__':
    sys.exit(test())
