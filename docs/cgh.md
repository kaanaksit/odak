# Computer-Generated Holography
Odak contains essential ingredients for research and development targeting Computer-Generated Holography.
We consult the beginners in this matter to `Goodman's Introduction to Fourier Optics` book (ISBN-13:  978-0974707723) and `Principles of optics: electromagnetic theory of propagation, interference and diffraction of light` from Max Born and Emil Wolf (ISBN 0-08-26482-4).
In the rest of this document, you will find engineering notes and relevant functions in Odak that helps you describing complex nature of light on a computer.

## Engineering notes

| Note          | Description   |
| ------------- |:-------------:|
| [`Holographic light transport`](notes/holographic_light_transport.md) | This engineering note will give you an idea about how coherent light propagates in free space. |
| [`Optimizing holograms using Odak`](notes/optimizing_holograms_using_odak.md) | This engineering note will give you an idea about how to calculate phase-only holograms using Odak. |

## odak.learn.wave
This submodule is based on `torch`, therefore the functions provided here are differentiable and can be used with provided optimizers in `torch`.

| Function      | Description   |
| ------------- |:-------------:|
| [odak.learn.wave.adjust_phase_only_slm_range](odak/learn/wave/adjust_phase_only_slm_range.md) | Adjust the range of a spatial light modulator. |
| [odak.learn.wave.band_limited_angular_spectrum](odak/learn/wave/band_limited_angular_spectrum.md) | Optical beam propagation with banlimited angular spectrum. |
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
| [odak.wave.wavenumber](odak/wave/wavenumber.md) | Wave number. |
