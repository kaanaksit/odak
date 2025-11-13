import sys
import odak
import torch
from tqdm import tqdm

def test(output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    final_surface_point = torch.tensor([0., 0., 10.])
    final_surface = odak.learn.raytracing.define_plane(point = final_surface_point)
    no = [500, 500]
    start_points, _, _, _ = odak.learn.tools.grid_sample(
                                                         no = no,
                                                         size = [10., 10.],
                                                         center = [0., 0., -10.]
                                                        )
    end_point = torch.tensor([0., 0., 0.])
    rays = odak.learn.raytracing.create_ray_from_two_points(start_points, end_point)
    mesh = odak.learn.raytracing.planar_mesh(
                                             size = torch.tensor([10., 10.]),
                                             number_of_meshes = torch.tensor([40, 40]),
                                             angles = torch.tensor([  0., -70., 0.]),
                                             offset = torch.tensor([ -2.,   0., 5.]),
                                            )
    triangles = mesh.get_triangles()
    play_button = torch.tensor([[
                                 [  1.,  0.5, 3.],
                                 [  0.,  0.5, 3.],
                                 [ 0.5, -0.5, 3.],
                                ]])
    triangles = torch.cat((play_button, triangles), dim = 0)
    background_color = torch.rand(3)
    triangles_color = torch.rand(triangles.shape[0], 3)
    image = torch.zeros(rays.shape[0], 3) 
    for triangle_id, triangle in enumerate(triangles):
        _, _, _, _, check = odak.learn.raytracing.intersect_w_triangle(rays, triangle)
        check = check.squeeze(0).unsqueeze(-1).repeat(1, 3)
        color = triangles_color[triangle_id].unsqueeze(0).repeat(check.shape[0], 1)
        image[check == True] = color[check == True] * check[check == True]
    image[image == [0., 0., 0]] = background_color
    image = image.view(no[0], no[1], 3)
    odak.learn.tools.save_image('{}/image.png'.format(output_directory), image, cmin = 0., cmax = 1.)
    assert True == True
      

if __name__ == '__main__':
    sys.exit(test())
