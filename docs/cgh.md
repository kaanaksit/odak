# Computer-Generated Holography
Odak contains essential ingredients for research and development targeting Computer-Generated Holography.
We consult the beginners in this matter to `Goodman's Introduction to Fourier Optics` book (ISBN-13:  978-0974707723) and `Principles of optics: electromagnetic theory of propagation, interference and diffraction of light` from Max Born and Emil Wolf (ISBN 0-08-26482-4).
In the rest of this document, you will find engineering notes and relevant functions in Odak that helps you describing complex nature of light on a computer.
Note that, the creators of this documentation are from `Computational Displays` domain, however the provided submodules can potentially aid other lines of research as well, such as `Computational Imaging` or `Computational Microscopy`.

## Engineering notes

| Note          | Description   |
| ------------- |:-------------:|
| [`Holographic light transport`](notes/holographic_light_transport.md) | This engineering note will give you an idea about how coherent light propagates in free space. |
| [`Optimizing phase-only single plane holograms using Odak`](notes/optimizing_holograms_using_odak.md) | This engineering note will give you an idea about how to calculate phase-only holograms using Odak. |
| [`Learning the model of a holographic display`](https://github.com/complight/realistic_holography) | This link navigates to a project website that provides a codebase that can learn the model of a holographic display using a single complex kernel. |

## odak.learn.wave
This submodule is based on `torch`, therefore the functions provided here are differentiable and can be used with provided optimizers in `torch`.

| Function      | Description   |
| ------------- |:-------------:|
| [odak.learn.wave.adjust_phase_only_slm_range](odak/learn/wave/adjust_phase_only_slm_range.md) | Adjust the range of a spatial light modulator. |
| [odak.learn.wave.band_limited_angular_spectrum](odak/learn/wave/band_limited_angular_spectrum.md) | Optical beam propagation with bandlimited angular spectrum. |
| [odak.learn.wave.calculate_phase](odak/learn/wave/calculate_phase.md) | Calculate phase of a complex field. |
| [odak.learn.wave.calculate_amplitude](odak/learn/wave/calculate_amplitude.md) | Calculate amplitude of a complex field. |
| [odak.learn.wave.custom](odak/learn/wave/custom.md)                 | Optical beam propagation with a custom complex kernel. |
| [odak.learn.wave.generate_complex_field](odak/learn/wave/generate_complex_field.md) | Generate a complex field from an amplitude and a phase value. |
| [odak.learn.wave.gerchberg_saxton](odak/learn/wave/gerchberg_saxton.md) | Phase-only hologram optimization using Gerchberg-Saxton algorithm. |
| [odak.learn.wave.impulse_response_fresnel](odak/learn/wave/impulse_response_fresnel.md) | Optical beam propagation with impulse response fresnel. |
| [odak.learn.wave.linear_grating](odak/learn/wave/linear_grating.md) | One or two axis linear grating.|
| [odak.learn.wave.point_wise](odak/learn/wave/point_wise.md) | Phase-only hologram optimization using point wise integration method. |
| [odak.learn.wave.prism_phase_function](odak/learn/wave/prism_phase_function.md) | Prism phase function, a prism.|
| [odak.learn.wave.produce_phase_only_slm_pattern](odak/learn/wave/produce_phase_only_slm_pattern.md) | Produce phase-only hologram for a spatial light modulator with given phase range. |
| [odak.learn.wave.propagate_beam](odak/learn/wave/propagate_beam.md) | General function for optical beam propagation. |
| [odak.learn.wave.quadratic_phase_function](odak/learn/wave/quadratic_phase_function.md) | Quadratic phase function, a lens.|
| [odak.learn.wave.set_amplitude](odak/learn/wave/set_amplitude.md) | Set amplitude of a complex field. |
| [odak.learn.wave.stochastic_gradient_descent](odak/learn/wave/stochastic_gradient_descent.md) | Phase-only hologram optimization using Stochastic Gradient Descent optimization. |
| [odak.learn.wave.transfer_function_kernel](odak/learn/wave/transfer_function_kernel.md) | Optical beam propagation with transfer function kernel. |
| [odak.learn.wave.wavenumber](odak/learn/wave/wavenumber.md) | Wave number. |


## odak.wave
This submodule is based on `Numpy`. This submodule existed much before `odak.learn.wave`. Though this submodule contains more functions than `odak.learn.wave`, it does not provide the flexibility in optimization provided by differentiation support of `torch`.

| Function      | Description   |
| ------------- |:-------------:|
| [odak.wave.adaptive_sampling_angular_spectrum](odak/wave/adaptive_sampling_angular_spectrum.md) | Propagate coherent beams with adaptive sampling angular spectrum method. |
| [odak.wave.angular_spectrum](odak/wave/angular_spectrum.md) | Optical beam propagation with angular spectrum. |
| [odak.wave.add_phase](odak/wave/add_phase.md) | Add a given phase to a given complex field. |
| [odak.wave.add_random_phase](odak/wave/add_random_phase.md) | Add a random phase to a given complex field. |
| [odak.wave.adjust_phase_only_slm_range](odak/wave/adjust_phase_only_slm_range.md) | Adjust the range of a spatial light modulator. |
| [odak.wave.band_extended_angular_spectrum](odak/wave/band_extended_angular_spectrum.md) | Optical beam propagation with band extended angular spectrum. |
| [odak.wave.band_limited_angular_spectrum](odak/wave/band_limited_angular_spectrum.md) | Optical beam propagation with bandlimited angular spectrum. |
| [odak.wave.calculate_amplitude](odak/wave/calculate_amplitude.md) | Calculate amplitude of a complex field. |
| [odak.wave.calculate_intensity](odak/wave/calculate_intensity.md) | Calculate intensity of a complex field. |
| [odak.wave.calculate_phase](odak/wave/calculate_phase.md) | Calculate phase of a complex field. |
| [odak.wave.double_convergence](odak/wave/double_convergence.md) | Provides an initial phase for beam shaping. |
| [odak.wave.electric_field_per_plane_wave](odak/wave/electric_field_per_plane_wave.md) | Return the state of a plane wave at a particular distance and time. |
| [odak.wave.fraunhofer_equal_size_adjust](odak/wave/fraunhofer_equal_size_adjust.md) | Match physical size of field with simulated. |
| [odak.wave.fraunhofer_inverse](odak/wave/fraunhofer_inverse.md) | Adjoint model for Fraunhofer beam propagation. |
| [odak.wave.generate_complex_field](odak/wave/generate_complex_field.md) | Generate a complex field from an amplitude and a phase value. |
| [odak.wave.gerchberg_saxton](odak/wave/gerchberg_saxton.md) | Phase-only hologram optimization using Gerchberg-Saxton algorithm. |
| [odak.wave.gerchberg_saxton_3d](odak/wave/gerchberg_saxton.md) | Phase-only hologram optimization using Gerchberg-Saxton algorithm for three-dimensional reconstructions. |
| [odak.wave.impulse_response_fresnel](odak/wave/impulse_response_fresnel.md) | Optical beam propagation with impulse response fresnel. |
| [odak.wave.linear_grating](odak/wave/linear_grating.md) | One or two axis linear grating.|
| [odak.wave.prism_phase_function](odak/wave/prism_phase_function.md) | Prism phase function, a prism. |
| [odak.wave.produce_phase_only_slm_pattern](odak/wave/produce_phase_only_slm_pattern.md) | Produce phase-only hologram for a spatial light modulator with given phase range. |
| [odak.wave.propagate_beam](odak/wave/propagate_beam.md) | General function for optical beam propagation. |
| [odak.wave.propagate_field](odak/wave/propagate_field.md) | Propagate a given array of spherical sources to given set of points in space. |
| [odak.wave.propagate_plane_waves](odak/wave/propagate_plane_waves.md) | Propagate a given plane wave in space and time. |
| [odak.wave.quadratic_phase_function](odak/wave/quadratic_phase_function.md) | Quadratic phase function, a lens.|
| [odak.wave.rayleigh_resolution](odak/wave/rayleigh_resolution.md) | Calculate Rayleigh resolution limit. |
| [odak.wave.rayleigh_sommerfeld](odak/wave/rayleigh_sommerfeld.md) | Optical beam propagation with Rayleigh-Sommerfeld integral. |
| [odak.wave.rotationspeed](odak/wave/rotationspeed.md) | Calculate rotation speed of a wave. |
| [odak.wave.set_amplitude](odak/wave/set_amplitude.md) | Set amplitude of a complex field. |
| [odak.wave.transfer_function_kernel](odak/wave/transfer_function_kernel.md) | Optical beam propagation with transfer function kernel. |
| [odak.wave.wavenumber](odak/wave/wavenumber.md) | Wave number. |
