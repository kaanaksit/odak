import torch
import odak


def test(
         m = 1,
         n = 1,
         resolution = [100, 100],
         visualize = True,
         subplot_titles = ['Real', 'Imaginary'],
         row_titles = [],
         title = 'Zernike Polynomials',
         device = torch.device('cpu'),
        ):

    x = torch.linspace(-1, 1, resolution[0], device = device)
    y = torch.linspace(-1, 1, resolution[1], device = device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    rho = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    theta = torch.atan2(grid_y, grid_x)
    row_titles = ['m: {}, n: {}'.format(m, n)]

    z_defocus = odak.learn.tools.zernike_polynomial(n, m, rho, theta)
    z_defocus_amplitude = odak.learn.wave.calculate_amplitude(z_defocus)
    z_defocus_phase = odak.learn.wave.calculate_phase(z_defocus)

    if visualize:
        diagram = odak.visualize.plotly.plot2dshow(
                                                   row_titles = row_titles,
                                                   title = title,
                                                   subplot_titles = subplot_titles,
                                                   rows = 1,
                                                   cols = 2,
                                                   shape = [1024, 600],
                                                  )
        diagram.add_field(
                          field = z_defocus_amplitude,
                          row = 1,
                          col = 1,
                          showscale = False,
                         )
        diagram.add_field(
                          field = z_defocus_phase,
                          row = 1,
                          col = 2,
                          showscale = False,
                         )
        diagram.show()
    assert True == True



if __name__ == "__main__":
    test()
