#!/usr/bin/env python


import sys
import odak
import torch
from tqdm import tqdm


def test():
    device = torch.device('cpu')
    ray_no = torch.tensor([150, 150], device = device)
    ray_size = [9., 9.]
    ray_start = [0., 0., 0.]
    ray_end = [0., 0., 10.]
    detector_size = torch.tensor([10., 10.], device = device)
    detector_resolution = torch.tensor([100, 100], device = device)
    detector_center = torch.tensor([0., 0., 0.], device = device)
    detector_tilt = torch.tensor([0., 0., 0.], device = device)
    detector_colors = 1
    mesh_size = torch.tensor([10., 10.], device = device)
    mesh_no = torch.tensor([30, 30], device = device)
    mesh_center = torch.tensor([0., 0., 10.], device = device)
    learning_rate = 2e-2
    number_of_steps = 200


    detector = odak.learn.raytracing.detector(
                                              colors = detector_colors,
                                              center = detector_center,
                                              tilt = detector_tilt,
                                              size = detector_size,
                                              resolution = detector_resolution,
                                              device = device
                                             )
    mesh = odak.learn.raytracing.planar_mesh(
                                             size = mesh_size,
                                             number_of_meshes = mesh_no,
                                             offset = mesh_center,
                                             device = device
                                            )
    start_points, _, _, _ = odak.learn.tools.grid_sample(
                                                         no = ray_no,
                                                         size = ray_size,
                                                         center = ray_start
                                                        )
    end_points, _, _, _ = odak.learn.tools.grid_sample(
                                                       no = ray_no,
                                                       size = ray_size,
                                                       center = ray_end
                                                      )
    start_points = start_points.to(device)
    end_points = end_points.to(device)
    rays = odak.learn.raytracing.create_ray_from_two_points(start_points, end_points)
    target = odak.learn.tools.load_image('test/kaan.png', normalizeby = 255., torch_style = True)[1].unsqueeze(0).to(device)
    optimizer = torch.optim.AdamW([mesh.heights], lr = learning_rate)
    t = tqdm(range(number_of_steps), leave = False, dynamic_ncols = True)
    #target_points, target_values = detector.convert_image_to_points(target, color = 0, ray_count = ray_no[0] * ray_no[1])
    #odak.learn.tools.save_image('target.png', target, cmin = 0., cmax = 1.)
    #from chamfer_distance import ChamferDistance as chamfer_dist
    #chd = chamfer_dist()
    #for step in t:
    #    optimizer.zero_grad()
    #    detector.clear()
    #    reflected_rays, _ = mesh.mirror(rays)
    #    points = detector.intersect(reflected_rays)
    #    image = detector.get_image()
    #    dist1, dist2, idx1, idx2 = chd(points.unsqueeze(0), target_points.unsqueeze(0))
    #    loss = (torch.sum(dist1)) + (torch.sum(dist2))
    #    loss.backward(retain_graph = True)
    #    optimizer.step()
    #    description = 'Loss: {}'.format(loss.item())
    #    t.set_description(description)
    #    if step % 10 == 0:
    #        odak.learn.tools.save_image('image.png', image, cmin = 0., cmax = image.max())
    #print(description)

    visualize = False
    if visualize:
        ray_diagram = odak.visualize.plotly.rayshow(
                                                    rows = 1,
                                                    columns = 1,
                                                    line_width = 3.,
                                                    marker_size = 1.,
                                                    subplot_titles = ['Optimization result']
                                                   )
        triangles = mesh.get_triangles()
        for triangle_id in range(triangles.shape[0]):
            ray_diagram.add_triangle(triangles[triangle_id], row = 1, column = 1, color = 'orange')
        html = ray_diagram.save_offline()
        markdown_file = open('ray.txt', 'w')
        markdown_file.write(html)
        markdown_file.close()
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
