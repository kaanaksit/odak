import sys
import odak
import torch # (1)


def test(directory = 'test_output'):
    odak.tools.check_directory(directory)
    starting_point = torch.tensor([[5., 5., 0.]]) # (2)
    end_points, _, _, _ = odak.learn.tools.grid_sample(
                                                       no = [2, 2], 
                                                       size = [20., 20.], 
                                                       center = [0., 0., 10.]
                                                      ) # (3)
    rays_from_points = odak.learn.raytracing.create_ray_from_two_points(
                                                                        starting_point,
                                                                        end_points
                                                                       ) # (4)


    starting_points, _, _, _ = odak.learn.tools.grid_sample(
                                                            no = [3, 3], 
                                                            size = [100., 100.], 
                                                            center = [0., 0., 10.],
                                                           )
    angles = torch.randn_like(starting_points) * 180. # (5)
    rays_from_angles = odak.learn.raytracing.create_ray(
                                                        starting_points,
                                                        angles
                                                       ) # (6)


    distances = torch.ones(rays_from_points.shape[0]) * 12.5
    propagated_rays = odak.learn.raytracing.propagate_ray(
                                                          rays_from_points,
                                                          distances
                                                         ) # (7)




    visualize = False # (8)
    if visualize:
        ray_diagram = odak.visualize.plotly.rayshow(line_width = 3., marker_size = 3.)
        ray_diagram.add_point(starting_point, color = 'red')
        ray_diagram.add_point(end_points[0], color = 'blue')
        ray_diagram.add_line(starting_point, end_points[0], color = 'green')
        x_axis = starting_point.clone()
        x_axis[0, 0] = end_points[0, 0]
        ray_diagram.add_point(x_axis, color = 'black')
        ray_diagram.add_line(starting_point, x_axis, color = 'black', dash = 'dash')
        y_axis = starting_point.clone()
        y_axis[0, 1] = end_points[0, 1]
        ray_diagram.add_point(y_axis, color = 'black')
        ray_diagram.add_line(starting_point, y_axis, color = 'black', dash = 'dash')
        z_axis = starting_point.clone()
        z_axis[0, 2] = end_points[0, 2]
        ray_diagram.add_point(z_axis, color = 'black')
        ray_diagram.add_line(starting_point, z_axis, color = 'black', dash = 'dash')
        html = ray_diagram.save_offline()
        markdown_file = open('{}/ray.txt'.format(directory), 'w')
        markdown_file.write(html)
        markdown_file.close()
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
