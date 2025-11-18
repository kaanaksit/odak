import sys
import odak
import torch


def visualize(
              model,
              points,
              ground_truth,
              sample_count = [20, 20, 20],
              threshold = 1e-1,
             ):
    diagram = odak.visualize.plotly.rayshow(
                                            columns = 3,
                                            line_width = 3.,
                                            marker_size = 3.,
                                            subplot_titles = [
                                                              'Gaussian Centers',
                                                              'Estimation',
                                                              'Ground truth',
                                                             ],
                                           )
    diagram.add_point(
                      model.centers.detach().cpu().numpy(), 
                      color = 'green',
                      column = 1,
                     )  

    x = torch.linspace(
                       torch.amin(points[:, 0]),
                       torch.amax(points[:, 0]),
                       sample_count[0], 
                       device = points.device
                      )
    y = torch.linspace(
                       torch.amin(points[:, 1]),
                       torch.amax(points[:, 1]),
                       sample_count[1], 
                       device = points.device
                      )
    z = torch.linspace(
                       torch.amin(points[:, 2]),
                       torch.amax(points[:, 2]),
                       sample_count[2], 
                       device = points.device
                      )
    X, Y, Z = torch.meshgrid(x, y, z, indexing = 'ij')
    samples = torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1), Z.unsqueeze(-1)), dim = -1)
    samples_flat = samples.reshape(-1, 3)

    estimate = model(samples_flat, test = True)
    estimate[estimate > 1.] = 1.
    estimate[estimate < 0.] = 0.

    diagram.add_volume(
                       points = samples_flat.detach().cpu().numpy(),
                       values = estimate.detach().cpu().numpy(),
                       limits = [1e-1, 1.],
                       surface_count = 27,
                       opacity = 0.3,
                       column = 2,
                      ) 
    points = points[ground_truth > threshold]
    ground_truth = ground_truth[ground_truth > threshold]
    diagram.add_point(
                      points.detach().cpu().numpy(), 
                      color = ground_truth.detach().cpu().numpy(),
                      column = 3,
                     ) 
    diagram.set_axis_limits(column = 1)
    diagram.set_axis_limits(column = 2)
    diagram.set_axis_limits(column = 3)
    diagram.show()


def get_training_data(
                      ply_filename,
                      voxel_size = [0.05, 0.05, 0.05],
                      device = torch.device('cpu'),
                     ):

    triangles = odak.tools.read_PLY(ply_filename)
    points = odak.raytracing.center_of_triangle(triangles)
    points = torch.as_tensor(points, device = device)
    points = points / 100 
    points[:, 0] += 0.10
    points[:, 1] -= 0.35
    points[:, 2] -= 0.20
    ground_truth = torch.ones(points.shape[0], device = device)
    voxel_locations, voxel_grid = odak.learn.tools.point_cloud_to_voxel(
                                                                        points = points,
                                                                        voxel_size = voxel_size,
                                                                       )
    points = voxel_locations.reshape(-1, 3)
    ground_truth = voxel_grid.reshape(-1)
    return points, ground_truth


def main(
         directory = 'test_output',
         ply_filename = './test/data/armadillo_low_poly.ply',
         number_of_elements = 300,
         learning_rate = 3e-2,
         number_of_epochs = 0, # Suggested: 10000,
         save_at_every = 1000,
         scheduler_power = 1,
         weights_filename = 'gaussian_3d_volume_weights.pt',
         device = torch.device('cpu'),
         visualization = False,
         loss_weights = {
                         'content' : {
                                      'l2'  : 1e+0,
                                      'l1'  : 0e-0,
                                     },
                         'alpha'   : {
                                      'sum' : 1e-2,
                                      'threshold' : [1e-2, 1000.]
                                     },
                         'scale'   : 1e-2,
                         'angle'   : 0e-0,
                         'center'  : 1e-2,
                        },
        ):
    odak.tools.check_directory(directory)
    weights_filename = '{}/{}'.format(directory, weights_filename)


    points, \
    ground_truth = get_training_data(
                                     ply_filename = ply_filename,
                                     device = device
                                    )


    model = odak.learn.models.gaussian_3d_volume(
                                                 number_of_elements = number_of_elements
                                                ).to(device)
    model.load_weights(
                       weights_filename = weights_filename,
                       device = device
                      )

    if number_of_epochs > 0:
        model.optimize(
                       points = points,
                       ground_truth = ground_truth,
                       loss_weights = loss_weights,
                       learning_rate = learning_rate,
                       number_of_epochs = number_of_epochs,
                       save_at_every = save_at_every,
                       scheduler_power = scheduler_power,
                       weights_filename = weights_filename,
                      )


    if visualization:
        visualize(
                  model = model,
                  points = points,
                  ground_truth = ground_truth,
                 )
    assert True == True


if __name__ == '__main__':
    sys.exit(main())
