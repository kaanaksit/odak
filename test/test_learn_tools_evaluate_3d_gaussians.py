import torch
import odak


def test(
         n = [20, 20, 20],
         limits = [
                   [-5, 5],
                   [-5, 5],
                   [-5, 5]
                  ],
         visualize = True,
         device = torch.device('cpu'),
        ):


    x = torch.linspace(limits[0][0], limits[0][1], n[0], device = device)
    y = torch.linspace(limits[1][0], limits[1][1], n[1], device = device)
    z = torch.linspace(limits[2][0], limits[2][1], n[2], device = device)
    X, Y = torch.meshgrid(x, y, indexing = 'ij')
    X = X.flatten().unsqueeze(-1)
    Y = Y.flatten().unsqueeze(-1)
    Z = torch.ones_like(X)
    points = torch.empty(0, 3, device = device)
    for i in z:
        new_points = torch.concat((X, Y, Z * i), dim = -1)
        points = torch.cat((points, new_points), dim = 0)


    centers = torch.tensor([
                            [3., 0., 0.],
                            [0., 3., 0.],
                            [-2., -2., 1.],
                            [0., 0., 4.],
                           ], 
                           device = device
                          )
    angles = torch.tensor([
                           [0., 10., 0.],
                           [0., 0., 0.],
                           [0., 0., 0.],
                           [0., 0., 0.],
                          ], 
                          device = device
                         ) 
    scales = torch.tensor([
                           [0.1, 1., 0.5],
                           [0.2, 0.2, 0.2],
                           [1., 2., 0.3],
                           [1., 1., 1.],
                          ],
                          device = device
                         )
    opacity = torch.tensor([[2.0, 1.1, 2.5, 20.]], device = device).T
                            


    intensities = odak.learn.tools.evaluate_3d_gaussians(
                                                         points = points,
                                                         centers = centers,
                                                         scales = scales,
                                                         angles = angles,
                                                         opacity = opacity,
                                                        )
    total_intensities = torch.sum(intensities, dim = -1)


    if visualize:
        points = points.cpu().numpy()
        centers = centers.cpu().numpy()
        total_intensities = total_intensities.cpu().numpy()
        total_intensities = total_intensities / total_intensities.max()
        ray_diagram = odak.visualize.plotly.rayshow(
                                                    line_width = 3.,
                                                    marker_size = 3.,
                                                    subplot_titles = ['Gaussians'],
                                                   )
        ray_diagram.add_volume(points, values = total_intensities, limits = [0., 1.], opacity = 0.3, surface_count = 40)
        ray_diagram.add_point(centers, color = 'green', opacity = 0.3)
        ray_diagram.show()
    assert True == True



if __name__ == "__main__":
    test()
