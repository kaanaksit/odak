import sys


def test():
    import numpy as np
    from odak.wave import gerchberg_saxton_3d, adjust_phase_only_slm_range, produce_phase_only_slm_pattern, calculate_amplitude, wavenumber, propagate_beam
    from odak.tools import save_image
    wavelength = 0.000000532
    dx = 0.0000064
    distances = np.array([0.2, 0.1])
    input_fields = np.random.rand(2, 500, 500).astype(np.complex64)
    iteration_number = 3
    distance_light_slm = 2.0
    k = wavenumber(wavelength)
    hologram = gerchberg_saxton_3d(
        input_fields,
        iteration_number,
        -distances,
        dx,
        wavelength,
        np.pi*2,
        'Bandlimited Angular Spectrum',
        initial_phase=None
    )
    reconstruction_0 = propagate_beam(
        hologram, k, distances[0], dx, wavelength, propagation_type='Bandlimited Angular Spectrum')
    reconstruction_1 = propagate_beam(
        hologram, k, distances[1], dx, wavelength, propagation_type='Bandlimited Angular Spectrum')
    amplitude_0 = calculate_amplitude(reconstruction_0)
    amplitude_1 = calculate_amplitude(reconstruction_1)
    hologram, _ = produce_phase_only_slm_pattern(
        hologram,
        2*np.pi,
        'output_hologram.png'
    )

    save_image(
        'output_amplitude_0.png',
        amplitude_0,
        cmin=0,
        cmax=np.amax(amplitude_0)
    )
    save_image(
        'output_amplitude_1.png',
        amplitude_1,
        cmin=0,
        cmax=np.amax(amplitude_1)
    )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
