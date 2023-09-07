import sys
import os
import odak
import numpy as np
from odak.wave import wavenumber, propagate_beam, add_random_phase


def test():
    wavelength = 515e-9
    pixeltom = 3.74e-6
    distance = 10e-4
    propagation_types = ['Impulse Response Fresnel', 'Transfer Function Fresnel']

    k = wavenumber(wavelength)
    sample_field = np.zeros((150, 150), dtype=np.complex64)
    sample_field = np.zeros((150, 150), dtype=np.complex64)
    sample_field[
                 sample_field.shape[0] // 2 - 10: sample_field.shape[0] // 2 + 10,
                 sample_field.shape[1] // 2 - 10: sample_field.shape[1] // 2 + 10
                ] += 1.
    for propagation_type in propagation_types:
         output_field = propagate_beam(
                                       sample_field,
                                       k,
                                       distance,
                                       pixeltom,
                                       wavelength,
                                       propagation_type
                                      )
         output_intensity = odak.wave.calculate_amplitude(output_field) ** 2
         odak.tools.save_image('{}.png'.format(propagation_type.replace(' ', '_')), output_intensity, cmin = 0., cmax = output_intensity.max())
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
