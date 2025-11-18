import sys
import odak
import torch


def visualize(
              model,
              points,
              ground_truth,
              threshold = 1e-1,
             ):
    estimate = model(points, test = True)
    estimate[estimate > 1.] = 1.
    estimate[estimate < 0.] = 0.
    thresholded_points = points[estimate > threshold]
    estimate = estimate[estimate > threshold]
    points = points[ground_truth > threshold]
    ground_truth = ground_truth[ground_truth > threshold]
    diagram = odak.visualize.plotly.rayshow(
                                            columns = 3,
                                            line_width = 3.,
                                            marker_size = 3.,
                                            subplot_titles = [
                                                              'Centers',
                                                              'Estimation',
                                                              'Ground truth',
                                                             ],
                                           )
    diagram.add_point(
                      model.centers.detach().cpu().numpy(), 
                      color = 'green',
                      column = 1,
                     ) 
    diagram.add_point(
                      thresholded_points.detach().cpu().numpy(), 
                      color = estimate.detach().cpu().numpy(),
                      column = 2,
                     ) 
    diagram.add_point(
                      points.detach().cpu().numpy(), 
                      color = ground_truth.detach().cpu().numpy(),
                      column = 3,
                     ) 
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
         learning_rate = 3e-3,
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
                         'sigma'   : 0e-1,
                         'alpha'   : 1e-1,
                         'angle'   : 0e-1,
                         'center'  : 0e-1,
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
