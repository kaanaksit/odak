import sys
import odak
import torch


def test():
    starting_points, _, _, _ = odak.learn.tools.grid_sample(
                                                            no = [5, 5],
                                                            size = [10., 10.],
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
    triangle1 = torch.tensor([[
                              [-5., -5., 10.],
                              [ 5., -5., 10.],
                              [ 0.,  5., 10.]
                            ]])
    
    triangle2 = torch.tensor([[
                              [-4., -4., 8.],
                              [ 4., -4., 8.],
                              [ 0.,  4., 8.]
                            ]])
    
    triangles = torch.concat([triangle1, triangle2])
    normals, distance, _, _, check = odak.learn.raytracing.intersect_w_triangle_batch(
                                                                                      rays,
                                                                                      triangles
                                                                                     )
    visualize = False
    if visualize:
        ray_diagram = odak.visualize.plotly.rayshow(line_width = 3., marker_size = 3.)
        for i, triangle in enumerate(triangles):
            ray_diagram.add_triangle(triangle, color = 'orange')
            ray_diagram.add_point(rays[:, 0], color = 'blue')
            ray_diagram.add_line(rays[:, 0], normals[i, :, 0], color = 'blue')
            colors = []
            for color_id in range(check.shape[1]):
                if check[i, color_id] == True:
                    colors.append('green')
                elif check[i, color_id] == False:
                    colors.append('red')
            ray_diagram.add_point(normals[i, :, 0], color = colors)
            #ray_diagram.show()

        html = ray_diagram.save_offline()
        markdown_file = open('ray.txt', 'w')
        markdown_file.write(html)
        markdown_file.close()
    assert True == True
   

if __name__ == '__main__':
    sys.exit(test())
