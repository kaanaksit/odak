import sys
import odak
import torch

def test(output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    starting_points, _, _, _ = odak.learn.tools.grid_sample(
                                                            no = [5, 5],
                                                            size = [15., 15.],
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
    n_air = 1.0 # (1)
    n_glass = 1.51 # (2)
    refracted_rays = odak.learn.raytracing.refract(intersecting_rays, intersecting_normals, n_air, n_glass) # (3)
    reflected_rays = odak.learn.raytracing.reflect(intersecting_rays, intersecting_normals) # (4)
    refract_distance = 11.
    reflect_distance = 7.2
    propagated_refracted_rays = odak.learn.raytracing.propagate_ray(
                                                                    refracted_rays, 
                                                                    torch.ones(refracted_rays.shape[0]) * refract_distance
                                                                   )
    propagated_reflected_rays = odak.learn.raytracing.propagate_ray(
                                                                    reflected_rays,
                                                                    torch.ones(reflected_rays.shape[0]) * reflect_distance
                                                                   )



    visualize = False
    if visualize:
        ray_diagram = odak.visualize.plotly.rayshow(
                                                    columns = 2,
                                                    line_width = 3.,
                                                    marker_size = 3.,
                                                    subplot_titles = ['Refraction example', 'Reflection example']
                                                   ) # (1)
        ray_diagram.add_triangle(triangle, column = 1, color = 'orange')
        ray_diagram.add_triangle(triangle, column = 2, color = 'orange')
        ray_diagram.add_point(rays[:, 0], column = 1, color = 'blue')
        ray_diagram.add_point(rays[:, 0], column = 2, color = 'blue')
        ray_diagram.add_line(rays[:, 0], normals[:, 0], column = 1, color = 'blue')
        ray_diagram.add_line(rays[:, 0], normals[:, 0], column = 2, color = 'blue')
        ray_diagram.add_line(refracted_rays[:, 0], propagated_refracted_rays[:, 0], column = 1, color = 'blue')
        ray_diagram.add_line(reflected_rays[:, 0], propagated_reflected_rays[:, 0], column = 2, color = 'blue')
        colors = []
        for color_id in range(check.shape[1]):
            if check[0, color_id] == True:
                colors.append('green')
            elif check[0, color_id] == False:
                colors.append('red')
        ray_diagram.add_point(normals[:, 0], column = 1, color = colors)
        ray_diagram.add_point(normals[:, 0], column = 2, color = colors)
        html = ray_diagram.save_offline()
        markdown_file = open('{}/ray.txt'.format(output_directory), 'w')
        markdown_file.write(html)
        markdown_file.close()
    assert True == True
   

if __name__ == '__main__':
    sys.exit(test())
