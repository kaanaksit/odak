import sys
import odak
import torch
from tqdm import tqdm


def test(output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    final_surface = torch.tensor([[
                                   [-5., -5., 0.],
                                   [ 5., -5., 0.],
                                   [ 0.,  5., 0.]
                                 ]])
    final_target = torch.tensor([[3., 3., 0.]])
    triangle = torch.tensor([
                             [-5., -5., 10.],
                             [ 5., -5., 10.],
                             [ 0.,  5., 10.]
                            ])
    starting_points, _, _, _ = odak.learn.tools.grid_sample(
                                                            no = [5, 5],
                                                            size = [1., 1.],
                                                            center = [0., 0., 0.]
                                                           )
    end_point = odak.learn.raytracing.center_of_triangle(triangle)
    rays = odak.learn.raytracing.create_ray_from_two_points(
                                                            starting_points,
                                                            end_point
                                                           )
    angles = torch.zeros(1, 3, requires_grad = True)
    learning_rate = 2e-1
    optimizer = torch.optim.Adam([angles], lr = learning_rate)
    loss_function = torch.nn.MSELoss()
    number_of_steps = 100
    t = tqdm(range(number_of_steps), leave = False, dynamic_ncols = True)
    for step in t:
        optimizer.zero_grad()
        rotated_triangle, _, _, _ = odak.learn.tools.rotate_points(
                                                                   triangle, 
                                                                   angles = angles, 
                                                                   origin = end_point
                                                                  )
        _, _, intersecting_rays, intersecting_normals, check = odak.learn.raytracing.intersect_w_triangle(
                                                                                                          rays,
                                                                                                          rotated_triangle
                                                                                                         )
        reflected_rays = odak.learn.raytracing.reflect(intersecting_rays, intersecting_normals)
        final_normals, _ = odak.learn.raytracing.intersect_w_surface(reflected_rays, final_surface)
        if step == 0:
            start_rays = rays.detach().clone()
            start_rotated_triangle = rotated_triangle.detach().clone()
            start_intersecting_rays = intersecting_rays.detach().clone()
            start_intersecting_normals = intersecting_normals.detach().clone()
            start_final_normals = final_normals.detach().clone()
        final_points = final_normals[:, 0]
        target = final_target.repeat(final_points.shape[0], 1)
        loss = loss_function(final_points, target)
        loss.backward(retain_graph = True)
        optimizer.step()
        t.set_description('Loss: {}'.format(loss.item()))
    print('Loss: {}, angles: {}'.format(loss.item(), angles))


    visualize = False
    if visualize:
        ray_diagram = odak.visualize.plotly.rayshow(
                                                    columns = 2,
                                                    line_width = 3.,
                                                    marker_size = 3.,
                                                    subplot_titles = [
                                                                       'Surace before optimization', 
                                                                       'Surface after optimization',
                                                                       'Hits at the target plane before optimization',
                                                                       'Hits at the target plane after optimization',
                                                                     ]
                                                   ) 
        ray_diagram.add_triangle(start_rotated_triangle, column = 1, color = 'orange')
        ray_diagram.add_triangle(rotated_triangle, column = 2, color = 'orange')
        ray_diagram.add_point(start_rays[:, 0], column = 1, color = 'blue')
        ray_diagram.add_point(rays[:, 0], column = 2, color = 'blue')
        ray_diagram.add_line(start_intersecting_rays[:, 0], start_intersecting_normals[:, 0], column = 1, color = 'blue')
        ray_diagram.add_line(intersecting_rays[:, 0], intersecting_normals[:, 0], column = 2, color = 'blue')
        ray_diagram.add_line(start_intersecting_normals[:, 0], start_final_normals[:, 0], column = 1, color = 'blue')
        ray_diagram.add_line(start_intersecting_normals[:, 0], final_normals[:, 0], column = 2, color = 'blue')
        ray_diagram.add_point(final_target, column = 1, color = 'red')
        ray_diagram.add_point(final_target, column = 2, color = 'green')
        html = ray_diagram.save_offline()
        markdown_file = open('{}/ray.txt'.format(output_directory), 'w')
        markdown_file.write(html)
        markdown_file.close()
    assert True == True
      

if __name__ == '__main__':
    sys.exit(test())
