import sys
import odak
import torch


def visualize(
              model,
              points,
              ground_truth,
              sample_count = [20, 20, 20],
              threshold = 1e-1,
              camera_locations = [
                                  [0., 0.7, -2.],
                                  [0., 0.7, -2.],
                                  [0., 0.7, -2.],
                                 ]
             ):
    diagram = odak.visualize.plotly.rayshow(
                                            columns = 3,
                                            line_width = 3.,
                                            marker_size = 3.,
                                            subplot_titles = [
                                                              '<b>Gaussian Centers</b> <br><b>Color:</b> Opacity',
                                                              '<b>Estimation</b>',
                                                              '<b>Ground truth</b>',
                                                             ],
                                           )
    centers = model.centers.detach().cpu().numpy()
    alphas = model.alphas.detach().squeeze(-1).detach().cpu().numpy()
    diagram.add_point(
                      centers, 
                      color = alphas, 
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
                       limits = [5e-2, 1.],
                       surface_count = 35,
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

    diagram.set_axis_limits(x_limits = [-1, 1], y_limits = [-1, 1], z_limits = [-1, 1], column = 1)
    diagram.set_axis_limits(x_limits = [-1, 1], y_limits = [-1, 1], z_limits = [-1, 1], column = 2)
    diagram.set_axis_limits(x_limits = [-1, 1], y_limits = [-1, 1], z_limits = [-1, 1], column = 3)

    for column_id in range(3):
        diagram.set_camera(
                           x = camera_locations[column_id][0], 
                           y = camera_locations[column_id][1],
                           z = camera_locations[column_id][2],
                           column = column_id + 1
                          )
    diagram.show()


def main(
         directory = 'test_output',
         ply_filename = './test/data/armadillo_low_poly.ply',
         ply_voxel_size = [5e-2, 5e-2, 5e-2],
         number_of_elements = 150,
         learning_rate = 1e-2,
         number_of_epochs = 0, #10000 suggested
         save_at_every = 1000,
         scheduler_power = 1,
         weights_filename = 'gaussian_3d_volume_weights.pt',
         device = torch.device('cpu'),
         visualization = False,
         loss_weights = {
                         'content'     : {
                                          'l2'  : 1e+0,
                                          'l1'  : 1e-3,
                                         },
                         'alpha'       : {
                                          'smaller'   : 1.0e-2,
                                          'larger'    : 0.,
                                          'threshold' : [1e-2, 1.]
                                         },
                         'scale'       : {
                                          'smaller'   : 1.0e-2,
                                          'larger'    : 0.,
                                          'threshold' : [1e-3, 1.]
                                         },
                         'angle'       : 0e-0,
                         'center'      : 1e-2,
                         'utilization' : {
                                          'l2'  : 0.0,
                                         }
                       }
        ):
    odak.tools.check_directory(directory)
    weights_filename = '{}/{}'.format(directory, weights_filename)


    points, \
    ground_truth = odak.learn.tools.load_voxelized_PLY(
                                                       ply_filename = ply_filename,
                                                       voxel_size = ply_voxel_size,
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
