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
                                             number_of_meshes = [20, 20],
                                             offset = torch.tensor([0., 0., 10.]),
                                             device = device
                                            )
    start_points, _, _, _ = odak.learn.tools.grid_sample(
                                                         no = [70, 70],
                                                         size = [9., 9.],
                                                         center = [0., 0., 0.]
                                                        )
    end_points, _, _, _ = odak.learn.tools.grid_sample(
                                                       no = [70, 70],
                                                       size = [9., 9.],
                                                       center = [0., 0., 10.]
                                                      )
    start_points = start_points.to(device)
    end_points = end_points.to(device)
    rays = odak.learn.raytracing.create_ray_from_two_points(start_points, end_points)
    assert True == True
    #target = torch.zeros(1, 100, 100, device = device)
    #target[0, 45:55, 45:55:] = torch.linspace(0., 1., 10)
    target = odak.learn.tools.load_image('test/k.png', normalizeby = 255., torch_style = True)[1].unsqueeze(0).to(device)
    learning_rate = 2e-2
    number_of_steps = 1000
    optimizer = torch.optim.AdamW([mesh.heights], lr = learning_rate)
    t = tqdm(range(number_of_steps), leave = False, dynamic_ncols = True)
    target_points, target_values = detector.convert_image_to_points(target)
    loss_function = torch.nn.MSELoss()
    odak.learn.tools.save_image('target.png', target, cmin = 0., cmax = 1.)
    for step in t:
        optimizer.zero_grad()
        detector.clear()
        reflected_rays, _ = mesh.mirror(rays)
        points = detector.intersect(reflected_rays)
        image = detector.get_image()
        distances = torch.sqrt(torch.sum((points.unsqueeze(1) - target_points.unsqueeze(0)) ** 2, dim = 2))
        loss_location = torch.sum(torch.min(distances, dim = 1).values)
        #loss_image = loss_function(image, target)
        #loss = loss_image #+ loss_location
        loss = loss_location
        loss.backward(retain_graph = True)
        optimizer.step()
        description = 'Loss: {}'.format(loss.item())
        t.set_description(description)
        if step % 10 == 0:
            odak.learn.tools.save_image('image.png', image, cmin = 0., cmax = image.max())
    print(description)
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
