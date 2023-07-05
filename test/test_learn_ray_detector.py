#!/usr/bin/env python


import sys
import odak
import torch
from tqdm import tqdm


def test():
    device = torch.device('cpu')
    detector_size = torch.tensor([0.1, 0.1], device = device)
    detector_resolution = torch.tensor([100, 100], device = device)
    detector_center = torch.tensor([0., 0., 0.], device = device)
    detector_tilt = torch.tensor([0., 0., 0.], device = device)
    detector_colors = 1
    mesh_size = torch.tensor([0.1, 0.1], device = device)
    mesh_no = torch.tensor([20, 20], device = device)
    mesh_center = torch.tensor([0., 0., 0.1], device = device)
    ray_no = torch.tensor([30, 30], device = device)
    ray_size = [0.095, 0.095]
    ray_start = [0., 0., 0.]
    ray_end = [0., 0., 0.1]
    learning_rate = 1e-4
    number_of_steps = 1
    save_at_every = 1
#    heights = odak.learn.tools.torch.load('test/heights.pt')
    heights = None
  

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
                                             device = device,
                                             heights = heights
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
#    target = torch.zeros(1, 100, 100, device = device)
#    target[0, 40:60, 40:60] = 1.
    optimizer = torch.optim.AdamW([mesh.heights], lr = learning_rate)
    t = tqdm(range(number_of_steps), leave = False, dynamic_ncols = True)
#    target_points, target_values = detector.convert_image_to_points(target, color = 0, ray_count = 10000)
#    target_points, target_values = detector.convert_image_to_points(target, color = 0, ray_count = ray_no[0] * ray_no[1])
#    target_points = torch.zeros(2, 3, device = device)
    odak.learn.tools.save_image('target.png', target, cmin = 0., cmax = 1.)
    loss_function = torch.nn.MSELoss(reduction = 'sum')
    for step in t:
        optimizer.zero_grad()
        detector.clear()
        reflected_rays, _ = mesh.mirror(rays)
        points = detector.intersect(reflected_rays)
        loss_hit = torch.abs(ray_no[0] * ray_no[1] - points.shape[0])
        image = detector.get_image()
#        distances = torch.sum((points.unsqueeze(1) - target_points.unsqueeze(0)) ** 2, dim = 2)
#        loss_distance = torch.sum(torch.min(distances, dim = 1).values)
#        distances = torch.sum((target_points.unsqueeze(1) - points.unsqueeze(0)) ** 2, dim = 2)
#        loss_rev_distance = torch.sum(torch.min(distances, dim = 1).values)
#        loss =  loss_distance + loss_hit + loss_rev_distance
        loss = loss_function(image, target)
        loss.backward(retain_graph = True)
        optimizer.step()
        description = 'Loss: {}'.format(loss.item())
        t.set_description(description)
        if step % save_at_every == 0:
            odak.learn.tools.save_image('image.png', image, cmin = 0., cmax = image.max())
            mesh.save_heights(filename = 'test/heights.pt')
            mesh.save_heights_as_PLY(filename = 'heights.ply')
    print(description)

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
