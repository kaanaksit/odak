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
    import torch
    from argparse import Namespace
    import odak


    def main():
        # 1. Define scene parameters.
        num_points = 128       # number of Gaussian primitives
        num_planes = 1         # number of hologram planes
        img_size = (64, 64)    # rendered image resolution (W, H)
        wavelengths = [633e-9] # red laser wavelength in metres

        args = Namespace(
            num_planes=num_planes,
            wavelengths=wavelengths,
            pixel_pitch=8e-6,           # 8 micron pixel pitch
            distances=[0.02],           # propagation distance to hologram plane
            pad_size=list(img_size),
            aperture_size=-1,           # no aperture
        )

        # 2. Create randomly initialised Gaussians.
        from odak.learn.wave.complex_gaussians import Gaussians, Scene
        gaussians = Gaussians(
            init_type="random",
            device="cpu",
            num_points=num_points,
            args_prop=args,
        )
        print(f"Initialised {len(gaussians)} Gaussians")

        # 3. Set up a perspective camera looking at the origin.
        from odak.learn.tools import PerspectiveCamera
        camera = PerspectiveCamera(
            R=torch.eye(3).unsqueeze(0),
            T=torch.tensor([[0.0, 0.0, 0.0]]),
            focal_length=torch.tensor([500.0, 500.0]),
            principal_point=torch.tensor([32.0, 32.0]),
        )

        # 4. Create a Scene and render the hologram.
        scene = Scene(gaussians, args)
        hologram, plane_field = scene.render(
            camera=camera,
            img_size=img_size,
            tile_size=(32, 32),
        )
        print(f"Hologram shape: {hologram.shape}, dtype: {hologram.dtype}")

        # 5. Extract amplitude and phase from the complex hologram.
        amplitude = odak.learn.wave.calculate_amplitude(hologram[0])
        phase = odak.learn.wave.calculate_phase(hologram[0])

        # 6. Visualise the Gaussian positions as a 3D point cloud.
        positions = gaussians.means.detach().cpu().numpy()
        colors = gaussians.colours.detach().cpu().numpy()

        visualize = False # (1)
        if visualize:
            diagram = odak.visualize.plotly.rayshow(
                columns=3,
                marker_size=3.0,
                subplot_titles=[
                    "<b>Gaussian positions</b>",
                    "<b>Hologram amplitude</b>",
                    "<b>Hologram phase</b>",
                ],
            )
            diagram.add_point(positions, color=colors, column=1)

            amplitude_np = amplitude.detach().cpu().numpy()
            phase_np = phase.detach().cpu().numpy()

            # Show amplitude and phase as heatmap-like 2D plots.
            import plotly.express as px
            fig_amp = px.imshow(amplitude_np, color_continuous_scale="hot", title="Amplitude")
            fig_phase = px.imshow(phase_np, color_continuous_scale="twilight", title="Phase")
            fig_amp.show()
            fig_phase.show()

            diagram.show()

        assert hologram.shape[0] == len(wavelengths)
        print("Done.")


    if __name__ == "__main__":
        sys.exit(main())
    ```

    1. Set `visualize = True` and install `plotly` (`pip install plotly`) to see the interactive 3D point cloud of Gaussian positions and the rendered hologram amplitude and phase.


The code above follows a simple pipeline:

1. **Define parameters** – number of Gaussians, image size, wavelength, pixel pitch, and propagation distance.
2. **Initialise Gaussians** – `Gaussians(init_type="random", ...)` creates randomly placed primitives with random colours, phases, and opacities.
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
