import sys
import odak
import torch

def test(output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    starting_points, _, _, _ = odak.learn.tools.grid_sample(
                                                            no = [5, 5],
                                                            size = [3., 3.],
                                                            center = [0., 0., 0.]
                                                           )
    end_points, _, _, _ = odak.learn.tools.grid_sample(
                                                       no = [5, 5],
                                                       size = [0.1, 0.1],
                                                       center = [0., 0., 5.]
                                                      )
    rays = odak.learn.raytracing.create_ray_from_two_points(
                                                            starting_points,
                                                            end_points
                                                           )
    center = torch.tensor([[0., 0., 5.]])
    radius = torch.tensor([[3.]])
    sphere = odak.learn.raytracing.define_sphere(
                                                 center = center,
                                                 radius = radius
                                                ) # (1)
    intersecting_rays, intersecting_normals, _, check = odak.learn.raytracing.intersect_w_sphere(rays, sphere)


    visualize = False # (2)
    if visualize:
        ray_diagram = odak.visualize.plotly.rayshow(line_width = 3., marker_size = 3.)
        ray_diagram.add_point(rays[:, 0], color = 'blue')
        ray_diagram.add_line(rays[:, 0][check == True], intersecting_rays[:, 0], color = 'blue')
        ray_diagram.add_sphere(sphere, color = 'orange')
        ray_diagram.add_point(intersecting_normals[:, 0], color = 'green')
        html = ray_diagram.save_offline()
        markdown_file = open('{}/ray.txt'.format(output_directory), 'w')
        markdown_file.write(html)
        markdown_file.close()
    assert True == True
   

if __name__ == '__main__':
    sys.exit(test())
