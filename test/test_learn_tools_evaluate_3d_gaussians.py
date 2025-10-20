import torch
import odak


def test(
         number_of_gaussians = 1,
         n = [50, 50, 3],
         limits = [
                   [-5, 5],
                   [-5, 5],
                   [-5, 5]
                  ],
         visualize = False,
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


    centers = torch.randn(number_of_gaussians, 3, device = device) * 3
    angles = torch.randn(number_of_gaussians, 3, device = device) * 180.
    sigmas = torch.rand(number_of_gaussians, 3, device = device) * 5


    intensities = odak.learn.tools.evaluate_3d_gaussians(
                                                         points = points,
                                                         centers = centers,
                                                         sigmas = sigmas,
                                                         angles = angles,
                                                        )
    total_intensities = torch.sum(intensities, dim = 0)


    if visualize:
        points = points.cpu().numpy()
        total_intensities = total_intensities.cpu().numpy()
        ray_diagram = odak.visualize.plotly.rayshow(
                                                    line_width = 3.,
                                                    marker_size = 3.,
                                                    subplot_titles = ['Gaussians'],
                                                   )
        ray_diagram.add_point(points, color = total_intensities)
        ray_diagram.show()
    assert True == True



if __name__ == "__main__":
    test()
