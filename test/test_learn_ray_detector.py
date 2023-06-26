#!/usr/bin/env python


import sys
import odak
import torch
from tqdm import tqdm


def test():
    device = torch.device('cpu')
    detector = odak.learn.raytracing.detector(
                                              colors = 1,
                                              center = torch.tensor([0., 0., 0.]),
                                              tilt = torch.tensor([0., 0., 0.]),
                                              size = torch.tensor([10., 10.]),
                                              resolution = torch.tensor([100, 100]),
                                              device = device
                                             )
    mesh = odak.learn.raytracing.planar_mesh(
                                             size = [10., 10.],
                                             number_of_meshes = [5, 5],
                                             offset = torch.tensor([0., 0., 10.]),
                                             device = device
                                            )
    start_points, _, _, _ = odak.learn.tools.grid_sample(
                                                         no = [5, 5],
                                                         size = [9., 9.],
                                                         center = [0., 5., 0.]
                                                        )
    end_points, _, _, _ = odak.learn.tools.grid_sample(
                                                       no = [5, 5],
                                                       size = [9., 9.],
                                                       center = [0., 0., 10.]
                                                      )
    start_points = start_points.to(device)
    end_points = end_points.to(device)
    rays = odak.learn.raytracing.create_ray_from_two_points(start_points, end_points)
    target = torch.rand(1, 9, 9)
    learning_rate = 2e-3
    number_of_steps = 100
    optimizer = torch.optim.AdamW([mesh.heights], lr = learning_rate)
    loss_function = torch.nn.MSELoss()
    t = tqdm(range(number_of_steps), leave = False, dynamic_ncols = True)
    assert True == True
    for step in t:
        optimizer.zero_grad()
        triangles = mesh.get_triangles()
        reflected_rays, _ = mesh.mirror(rays)
        image = detector.intersect(reflected_rays)
        import sys;sys.exit()
        loss = loss_function(image, target)
        loss.backward(retain_graph = True)
        optimizer.step()
        description = 'Loss: {}'.format(loss.item())
        t.set_description(description)
    print(description)
    assert True == True
      

if __name__ == '__main__':
    sys.exit(test())
