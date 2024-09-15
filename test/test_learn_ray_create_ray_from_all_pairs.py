import sys
import odak
import torch


def test(directory = 'test_output'):
    odak.tools.check_directory(directory)
    starting_points, _, _, _ = odak.learn.tools.grid_sample(
                                                            no = [2, 2], 
                                                            size = [20., 20.], 
                                                            center = [0., 0., 0.]
                                                           )
    
    end_points, _, _, _ = odak.learn.tools.grid_sample(
                                                       no = [2, 2], 
                                                       size = [20., 20.], 
                                                       center = [0., 0., 10.]
                                                      ) 
    
    rays_from_points_to_points = odak.learn.raytracing.create_ray_from_all_pairs(starting_points, end_points)

    distances = torch.ones(rays_from_points_to_points.shape[0]) * 12.5
    propagated_rays = odak.learn.raytracing.propagate_ray(
                                                          rays_from_points_to_points,
                                                          distances
                                                         )


    visualize = False
    if visualize:
        ray_diagram = odak.visualize.plotly.rayshow(line_width = 3., marker_size = 3.)
        ray_diagram.add_point(starting_points, color = 'red')
        ray_diagram.add_point(propagated_rays[:, 0], color = 'blue')

        ray_diagram.add_line(rays_from_points_to_points[:, 0], propagated_rays[:, 0])

        html = ray_diagram.save_offline()
        markdown_file = open('{}/ray.txt'.format(directory), 'w')
        markdown_file.write(html)
        markdown_file.close()
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
