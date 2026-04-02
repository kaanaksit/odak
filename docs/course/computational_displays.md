https://github.com/IdleHandsProject/volumetric_display

??? question end "Is there a good resource for classifying existing Augmented Reality glasses?"
    Using your favorite search engine, investigate if there is a reliable up-to-date table that helps comparing existing Augmented Reality glasses in terms of functionality and technical capabilities (e.g., field-of-View, resolution, focus cues).


## Complex-valued Gaussian splatting for holography


:octicons-info-24: Informative ·
:octicons-beaker-24: Practical


Traditional 3D Gaussian Splatting represents a scene as a collection of 3D Gaussian primitives, each with a mean position, covariance (orientation and scale), colour, and opacity.
These Gaussians are "splatted" onto the image plane by projecting their 3D covariance into 2D and alpha-compositing them in depth order to produce a rendered image.
Complex-valued Gaussian splatting extends this idea to holographic rendering.
Instead of compositing real-valued colours, each Gaussian carries a complex amplitude and a phase.
When splatted, the contributions are summed as complex fields, and the resulting field is propagated to the hologram plane using the band-limited angular spectrum method.
This produces a complex hologram that encodes both amplitude and phase information suitable for driving a holographic display.

`odak` provides a pure PyTorch implementation of this pipeline in `odak.learn.wave.complex_gaussians`, free of any external dependencies beyond PyTorch itself.
The key classes are:

- **`Gaussians`**: Stores and manages the 3D Gaussian primitives (means, quaternion rotations, scales, colours, phases, opacities, and plane assignments).
- **`Scene`**: Combines a set of `Gaussians` with a camera to render complex holograms via tile-based splatting and angular spectrum propagation.
- **`PerspectiveCamera`**: A lightweight camera model available from `odak.learn.tools` that stores rotation, translation, focal length, and principal point.


??? question end "How does complex Gaussian splatting differ from standard Gaussian splatting?"
    In standard 3D Gaussian splatting, Gaussians are alpha-composited to produce real-valued pixel colours.
    In the complex-valued variant, each Gaussian contributes a complex field $A \cdot e^{i\phi}$, where $A$ is the amplitude (derived from colour and opacity) and $\phi$ is a learned phase.
    The key equations remain similar for projection (3D covariance to 2D covariance), but the rendering step sums complex contributions rather than blending colours.
    The summed complex field is then propagated to the hologram plane using band-limited angular spectrum propagation to generate a hologram.


??? question end "What is band-limited angular spectrum propagation?"
    The angular spectrum method propagates a 2D complex field by a distance $d$ through free space.
    In the frequency domain, propagation amounts to multiplying the field's Fourier transform by a transfer function:

    $$
    H(f_x, f_y) = \exp\left(i \cdot d \cdot \sqrt{k^2 - (2\pi f_x)^2 - (2\pi f_y)^2}\right),
    $$

    where $k = 2\pi / \lambda$ is the wavenumber.
    The "band-limited" variant applies a frequency mask to avoid aliasing artefacts from evanescent waves, ensuring physically accurate results.


### Usage example

The script below demonstrates how to initialise a set of random complex Gaussians, set up a camera, render a hologram, and visualise the result.
We keep the example brief so that first-time readers can follow each step.


=== ":octicons-file-code-16: `test_learn_wave_complex_gaussians.py`"

    ```python
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

    ```

    1. All Gaussian parameters are explicitly overridden here for a controlled demo. **Positions** are placed adaptively based on the image resolution: from the pinhole model `u = fx·X/Z + px`, the half-extent is `(coverage/2) · img_size · z_depth / focal_length`, so the Gaussians always cover 60% of the image regardless of resolution. The `principal_point` is derived from `img_size` (`img_size[0]/2, img_size[1]/2`) so the projection is always centred. **Colours** are uniform random in [0, 1). **Rotations** are fully random quaternions (normalized at render time), giving varied orientations so the projected Gaussians appear stretched rather than circular. **Scales** are independent per axis (anisotropic) in log-space, so each Gaussian has different x/y/z extents. **Phases** are standard-normal (wrapped to [0, 2π) at render time). **Opacities** are set to logit 1.0, i.e. `sigmoid(1.0) ≈ 0.73`. **Plane assignments** use high-variance logits (σ = 10) for near-deterministic one-hot assignment via the Straight-Through Estimator.


The code above follows a simple pipeline:

1. **Define parameters** – number of Gaussians, image size, wavelength, pixel pitch, and propagation distance.
2. **Initialise Gaussians** – `Gaussians(init_type="random", ...)` creates primitives, then all seven parameter groups are explicitly overridden: positions (means), colours, rotations (quaternions), scales, phases, opacities, and plane assignments.
3. **Set up the camera** – `PerspectiveCamera` from `odak.learn.tools` defines the view with rotation, translation, focal length, and principal point.
4. **Render** – `Scene.render()` performs depth-sorted tile-based splatting followed by band-limited angular spectrum propagation to produce a complex hologram.
5. **Analyse** – `odak.learn.wave.calculate_amplitude` and `odak.learn.wave.calculate_phase` extract the amplitude and phase from the complex field.

Let us also examine the key classes provided in `odak` for this pipeline.


=== ":octicons-file-code-16: `odak.learn.wave.complex_gaussians.Gaussians`"

    ::: odak.learn.wave.complex_gaussians.Gaussians

=== ":octicons-file-code-16: `odak.learn.wave.complex_gaussians.Scene`"

    ::: odak.learn.wave.complex_gaussians.Scene

=== ":octicons-file-code-16: `odak.learn.tools.PerspectiveCamera`"

    ::: odak.learn.tools.PerspectiveCamera


??? question end "Can I load Gaussians from a trained checkpoint instead of random initialisation?"
    Yes. Use `init_type="gaussians"` with a path to a `.pth` checkpoint:

    ```python
    gaussians = Gaussians(
        init_type="gaussians",
        device="cuda",
        load_path="path/to/checkpoint.pth",
        args_prop=args,
    )
    ```

    The checkpoint should contain the keys `means`, `pre_act_quats`, `pre_act_scales`, `colours`, `pre_act_phase`, `pre_act_opacities`, and `pre_act_plane_assignment`.


??? question end "Can I initialise from a point cloud?"
    Yes. Use `init_type="point"` with a dictionary containing `positions` and `colors` tensors:

    ```python
    pointcloud_data = {
        "positions": positions_tensor,  # (N, 3)
        "colors": colors_tensor,        # (N, 3)
    }
    gaussians = Gaussians(
        init_type="point",
        device="cpu",
        pointcloud_data=pointcloud_data,
        args_prop=args,
    )
    ```
