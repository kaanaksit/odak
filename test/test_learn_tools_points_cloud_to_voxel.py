import sys
import odak
import torch


def visualize(
    points,
    values,
):
    diagram = odak.visualize.plotly.rayshow(
        columns=1,
        line_width=3.0,
        marker_size=3.0,
        subplot_titles=["Voxalized PLY model"],
    )
    diagram.add_point(
        points.detach().cpu().numpy(),
        color=values.detach().cpu().numpy(),
        column=1,
    )
    diagram.show()


def test(
    voxel_size=[0.07, 0.07, 0.07],
    ply_filename="./test/data/armadillo_low_poly.ply",
    visualization=False,
    device=torch.device("cpu"),
):
    triangles = odak.tools.read_PLY(ply_filename)
    points = odak.raytracing.center_of_triangle(triangles)
    points = torch.as_tensor(points, device=device)
    points = points / 100
    voxel_locations, voxel_grid = odak.learn.tools.point_cloud_to_voxel(
        points=points,
        voxel_size=voxel_size,
    )
    points = voxel_locations.reshape(-1, 3)
    values = voxel_grid.reshape(-1)

    if visualization:
        visualize(
            points=points,
            values=values,
        )
    assert True == True


if __name__ == "__main__":
    sys.exit(test())
