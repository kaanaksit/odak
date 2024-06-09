import sys
import numpy as np
import odak
from odak.wave import gerchberg_saxton, adjust_phase_only_slm_range, produce_phase_only_slm_pattern, calculate_amplitude, wavenumber, double_convergence
from odak.tools import save_image


def test(output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    wavelength = 532e-9
    dx = 8e-6
    distance = 2.0
    input_field = np.zeros((500, 500), dtype=np.complex64)
    input_field[0::50, :] += 1
    iteration_number = 2
    distance_light_slm = 2.0
    k = wavenumber(wavelength)
    initial_phase = double_convergence(
                                       input_field.shape[0],
                                       input_field.shape[1],
                                       k,
                                       distance + distance_light_slm,
                                       dx
                                      )
    hologram, reconstructed = gerchberg_saxton(
                                               input_field,
                                               iteration_number,
                                               distance,
                                               dx,
                                               wavelength,
                                               np.pi * 2,
                                               'Bandlimited Angular Spectrum',
                                               initial_phase = initial_phase
                                              )
    hologram, _ = produce_phase_only_slm_pattern(
                                                 hologram,
                                                 2 * np.pi,
                                                 '{}/output_hologram.png'.format(output_directory)
                                                )
    amplitude = calculate_amplitude(reconstructed)
    save_image(
               '{}/output_amplitude.png'.format(output_directory),
               amplitude,
               cmin = 0,
               cmax = np.amax(amplitude)
              )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
