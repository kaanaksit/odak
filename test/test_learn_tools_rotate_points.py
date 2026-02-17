import sys
import odak
import torch
from tqdm import tqdm


def test(
    output_directory="test_output",
    visualization=False,
    device=torch.device("cpu"),
    header="test_learn_tools_rotate_points.py",
):
    odak.tools.check_directory(output_directory)
    points = torch.tensor(
        [
            [10.0, 0.0, 5.0],
            [5.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ],
        device=device,
    )
    origins = torch.tensor(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0],
        ],
        device=device,
    )
    angles = torch.tensor(
        [
            [0.0, 0.0, -90.0],
            [-45.0, 0.0, 0.0],
        ],
        device=device,
    )
    odak.log.logger.info("{} -> Input points: {}".format(header, points))
    rotated_points, _, _, _ = odak.learn.tools.rotate_points(
        points,
        angles=angles,
        origin=origins,
    )
    ground_truth = torch.tensor(
        [
            [[-4.3711e-07, -1.0000e01, 5.0000e00], [1.0000e01, 3.5355e00, 3.5355e00]],
            [[5.0000e00, -5.0000e00, 0.0000e00], [5.0000e00, 3.5355e00, -3.5355e00]],
            [[0.0000e00, 0.0000e00, 5.0000e00], [0.0000e00, 3.5355e00, 3.5355e00]],
        ],
        device=device,
    )
    odak.log.logger.info("{} -> Rotated points: {}".format(header, rotated_points))
    test_result = torch.allclose(rotated_points, ground_truth, rtol=1e-4)

    if visualization:
        if len(rotated_points.shape) == 2:
            rotated_points = rotated_points.unsqueeze(-1)
        rotated_points = rotated_points.detach().cpu().numpy()
        origins = origins.detach().cpu().numpy()
        points = points.detach().cpu().numpy()
        titles = []
        for i in range(rotated_points.shape[0]):
            for j in range(rotated_points.shape[1]):
                title = "Point <b>{}</b> rotated around <b>{}</b><br> with <b>{}</b> degrees:<br><b>{}</b>".format(
                    points[i], origins[j], angles[j], rotated_points[i, j]
                )
                titles.append(title)
        diagram = odak.visualize.plotly.rayshow(
            rows=rotated_points.shape[0],
            columns=rotated_points.shape[1],
            line_width=3.0,
            marker_size=3.0,
            subplot_titles=titles,
        )
        for i in range(rotated_points.shape[0]):
            for j in range(rotated_points.shape[1]):
                diagram.add_point(
                    origins[j],
                    color="black",
                    opacity=0.3,
                    row=i + 1,
                    column=j + 1,
                )
                diagram.add_point(
                    points[i],
                    color="red",
                    opacity=0.3,
                    row=i + 1,
                    column=j + 1,
                )
                diagram.add_point(
                    rotated_points[i, j],
                    color="green",
                    opacity=0.3,
                    row=i + 1,
                    column=j + 1,
                )
                diagram.add_line(
                    rotated_points[i, j],
                    origins[j],
                    color="black",
                    row=i + 1,
                    column=j + 1,
                )
                diagram.add_line(
                    points[i],
                    origins[j],
                    color="black",
                    row=i + 1,
                    column=j + 1,
                )
        diagram.show()
    assert test_result == True


if __name__ == "__main__":
    sys.exit(test())
