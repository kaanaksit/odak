import sys
import odak
import torch
from tqdm import tqdm


def test(output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    device = torch.device('cpu')
    detector_size = torch.tensor([0.1, 0.1], device = device)
    detector_resolution = torch.tensor([100, 100], device = device)
    detector_center = torch.tensor([0., 0., 0.], device = device)
    detector_tilt = torch.tensor([0., 0., 0.], device = device)
    detector_colors = 1
    mesh_size = torch.tensor([0.1, 0.1], device = device)
    mesh_no = torch.tensor([20, 20], device = device)
    mesh_center = torch.tensor([0., 0., 0.1], device = device)
    ray_no = torch.tensor([40, 40], device = device)
    ray_size = [0.095, 0.095]
    ray_start = [0., 0., 0.]
    ray_end = [0., 0., 0.1]
    learning_rate = 4e-5
    number_of_steps = 1
    save_at_every = 1
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
    target = odak.learn.tools.load_image('test/data/kaan.png', normalizeby = 255., torch_style = True)[1].unsqueeze(0).to(device)
    target_binary = torch.ones_like(target) * (target > 0.) * 1.
    target_binary = target_binary.reshape(1, -1)
    target_binary_inverted = torch.abs(1. - target_binary) * 1e6
    optimizer = torch.optim.AdamW([mesh.heights,], lr = learning_rate)
    t = tqdm(range(number_of_steps), leave = False, dynamic_ncols = True)
    odak.learn.tools.save_image('{}/ray_target.png'.format(output_directory), target, cmin = 0., cmax = 1.)
    loss_function = torch.nn.MSELoss(reduction = 'sum')
    mu = 1e-6
    for step in t:
        optimizer.zero_grad()
        detector.clear()
        reflected_rays, _ = mesh.mirror(rays)
        points, values, distance_image = detector.intersect(reflected_rays)
        distance_min  = torch.min(distance_image, dim = 1).values.unsqueeze(-1)
        target_locations = torch.sum(1. / mu / torch.sqrt(torch.tensor(2 * odak.pi)) * torch.exp(- (distance_image - distance_min) ** 2 / 2. / mu ** 2), dim = 0)
        target_locations = target_locations.reshape(1, 100, 100) / target_locations.max()
        loss = loss_function(target_locations, target)
        image = detector.get_image()
        loss.backward(retain_graph = True)
        optimizer.step()
        description = 'Loss: {}'.format(loss.item())
        t.set_description(description)
        if step % save_at_every == 0:
            odak.learn.tools.save_image('{}/image_{:04d}.png'.format(output_directory, step), image, cmin = 0., cmax = image.max())
            odak.learn.tools.save_image('{}/targets_{:04d}.png'.format(output_directory, step), target_locations, cmin = 0., cmax = image.max())
            mesh.save_heights(filename = '{}/heights.pt'.format(output_directory))
            mesh.save_heights_as_PLY(filename = '{}/heights.ply'.format(output_directory))
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
