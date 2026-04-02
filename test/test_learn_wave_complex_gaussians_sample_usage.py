import sys
from argparse import Namespace

import numpy as np
import odak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


def main():
    # 1. Define scene parameters.
    num_points = 256
    distances=[0.03, 0.02, 0.01]
    num_planes=len(distances)
    img_size = (512, 512)
    focal_length = (500.0, 500.0)
    z_depth = 2.0
    coverage = 0.6
    wavelengths = [633e-9, 532e-9, 450e-9]

    args = Namespace(
        num_planes=num_planes,
        wavelengths=wavelengths,
        pixel_pitch=8e-6,
        distances=distances,
        pad_size=list(img_size),
        aperture_size=-1,
    )

    # 2. Create randomly initialised Gaussians.
    from odak.learn.wave.complex_gaussians import Gaussians, Scene

    gaussians = Gaussians(
        init_type="random",
        device="cpu",
        num_points=num_points,
        args_prop=args,
    )

    with torch.no_grad():
        # Positions: adaptive to resolution so Gaussians cover 60% of the image.
        # From pinhole model u = fx*X/Z + px, the half-extent for the desired
        # coverage is: x_half = (coverage/2) * img_size[0] * z_depth / fx.
        x_half = (coverage / 2) * img_size[0] * z_depth / focal_length[0]
        y_half = (coverage / 2) * img_size[1] * z_depth / focal_length[1]
        gaussians.means.data[:, 0] = torch.rand(num_points) * 2 * x_half - x_half
        gaussians.means.data[:, 1] = torch.rand(num_points) * 2 * y_half - y_half
        gaussians.means.data[:, 2] = z_depth + torch.rand(num_points) * 0.2

        # Colours / amplitude weights: random RGB in [0, 1) per channel.
        gaussians.colours.data = torch.rand(num_points, 3)

        # Rotations: random quaternions for varied orientations.
        # Normalized to unit quaternion at render time via F.normalize().
        gaussians.pre_act_quats.data = torch.randn(num_points, 4)

        # Scales (log-space): independent per axis for anisotropic (stretched)
        # Gaussians. exp() is applied at render time.
        gaussians.pre_act_scales.data = torch.log(
            torch.rand(num_points, 3) * 0.03 + 1e-6
        )

        # Phases: random per-channel, wrapped to [0, 2*pi) at render time.
        gaussians.pre_act_phase.data = torch.randn(num_points, 3)

        # Opacities (logit-space): sigmoid(1.0) ≈ 0.73 opacity.
        gaussians.pre_act_opacities.data = torch.ones(num_points)

        # Plane assignment logits: high variance for near-deterministic
        # one-hot assignment via StraightThroughEstimator at render time.
        gaussians.pre_act_plane_assignment.data = (
            torch.randn(num_points, num_planes) * 10.0
        )

    print(f"Initialised {len(gaussians)} Gaussians")

    # 3. Set up a perspective camera.
    from odak.learn.tools import PerspectiveCamera

    camera = PerspectiveCamera(
        R=torch.eye(3).unsqueeze(0),
        T=torch.tensor([[0.0, 0.0, 0.0]]),
        focal_length=torch.tensor(list(focal_length)),
        principal_point=torch.tensor([img_size[0] / 2.0, img_size[1] / 2.0]),
    )

    # 4. Render.
    scene = Scene(gaussians, args)
    hologram, plane_field = scene.render(
        camera=camera,
        img_size=img_size,
        tile_size=(32, 32),
    )
    print(f"Hologram shape: {hologram.shape}, dtype: {hologram.dtype}")

    # 5. Visualise the results.
    positions = gaussians.means.detach().cpu().numpy()

    # Plane index per Gaussian (argmax of assignment logits).
    plane_ids = gaussians.pre_act_plane_assignment.data.argmax(dim=1)

    # Distinct colours for each plane (works for N planes).
    plane_colors = [
        "red", "blue", "green", "orange", "purple",
        "cyan", "magenta", "yellow", "lime", "pink",
    ]

    # 3D point cloud of Gaussian positions, coloured by plane index.
    diagram = odak.visualize.plotly.rayshow(
        columns=1,
        marker_size=5.0,
        subplot_titles=["<b>Gaussian positions (colour = plane index)</b>"],
    )
    for p in range(num_planes):
        mask = (plane_ids == p).numpy()
        if mask.any():
            color = plane_colors[p % len(plane_colors)]
            diagram.add_point(
                positions[mask],
                color=color,
                column=1,
                show_legend=True,
                label=f"Plane {p}",
            )
    diagram.show()

    def complex_field_to_rgb(field_chw):
        """Convert (C, H, W) complex field to amplitude and phase RGB images (H, W, 3) uint8."""
        amp = np.abs(field_chw)
        phase = np.angle(field_chw)
        # Normalize amplitude to [0, 255].
        a_min, a_max = amp.min(), amp.max()
        if a_max > a_min:
            amp = (amp - a_min) / (a_max - a_min)
        amp_rgb = (amp.transpose(1, 2, 0) * 255).astype(np.uint8)
        # Normalize phase from [-pi, pi] to [0, 255].
        phase = (phase + np.pi) / (2 * np.pi)
        phase_rgb = (phase.transpose(1, 2, 0) * 255).astype(np.uint8)
        return amp_rgb, phase_rgb

    # Full-colour RGB images: one row per plane field + one row for the final hologram.
    total_rows = num_planes + 1

    subplot_titles = []
    row_titles = []
    for p in range(num_planes):
        subplot_titles.extend([f"Plane {p} Amplitude", f"Plane {p} Phase"])
        row_titles.append(f"Plane {p} (d={args.distances[p]} m)")
    subplot_titles.extend(["Hologram Amplitude", "Hologram Phase"])
    row_titles.append("Final Hologram")

    fig = make_subplots(
        rows=total_rows,
        cols=2,
        subplot_titles=subplot_titles,
        row_titles=row_titles,
        vertical_spacing=0.02,
        horizontal_spacing=0.02,
    )
    for p in range(num_planes):
        field = plane_field[p].detach().cpu().numpy()
        amp_rgb, phase_rgb = complex_field_to_rgb(field)
        fig.add_trace(go.Image(z=amp_rgb), row=p + 1, col=1)
        fig.add_trace(go.Image(z=phase_rgb), row=p + 1, col=2)
    holo = hologram.detach().cpu().numpy()
    amp_rgb, phase_rgb = complex_field_to_rgb(holo)
    fig.add_trace(go.Image(z=amp_rgb), row=total_rows, col=1)
    fig.add_trace(go.Image(z=phase_rgb), row=total_rows, col=2)
    fig.update_layout(width=1200, height=600 * total_rows)
    fig.show()

    assert hologram.shape[0] == len(wavelengths)
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
