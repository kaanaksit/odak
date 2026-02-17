import torch
import odak


def test(
    resolution=[100, 100],
    visualize=True,
    subplot_titles=[],
    row_titles=[],
    title="Zernike Polynomials",
    device=torch.device("cpu"),
):

    x = torch.linspace(-1, 1, resolution[0], device=device)
    y = torch.linspace(-1, 1, resolution[1], device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    rho = torch.sqrt(grid_x**2 + grid_y**2)
    theta = torch.atan2(grid_y, grid_x)

    z_piston = odak.learn.tools.zernike_polynomial(n=0, m=0, rho=rho, theta=theta)
    z_tilt_x = odak.learn.tools.zernike_polynomial(n=1, m=1, rho=rho, theta=theta)
    z_tilt_y = odak.learn.tools.zernike_polynomial(n=1, m=-1, rho=rho, theta=theta)
    z_defocus = odak.learn.tools.zernike_polynomial(n=2, m=0, rho=rho, theta=theta)
    z_comma_x = odak.learn.tools.zernike_polynomial(n=3, m=1, rho=rho, theta=theta)
    z_comma_y = odak.learn.tools.zernike_polynomial(n=3, m=-1, rho=rho, theta=theta)
    z_spherical = odak.learn.tools.zernike_polynomial(n=4, m=0, rho=rho, theta=theta)

    if visualize:
        fields = [
            z_piston,
            z_tilt_x,
            z_tilt_y,
            z_defocus,
            z_comma_x,
            z_comma_y,
            z_spherical,
        ]
        fields_len = len(fields)
        row_titles = [
            "piston",
            "tilt-x",
            "tilt-y",
            "defocus",
            "comma-x",
            "comma-y",
            "spherical",
        ]
        diagram = odak.visualize.plotly.plot2dshow(
            title=title,
            row_titles=row_titles,
            subplot_titles=subplot_titles,
            rows=fields_len,
            cols=1,
            shape=[512, 600 * fields_len],
        )
        for field_id, field in enumerate(fields):
            diagram.add_field(
                field=field,
                row=field_id + 1,
                col=1,
                showscale=False,
            )
        diagram.show()
    assert True == True


if __name__ == "__main__":
    test()
