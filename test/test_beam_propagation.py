import sys
import os
import odak
import numpy as np
from odak.wave import wavenumber, propagate_beam, add_random_phase


def test():
    # Variables to be set.
    wavelength = 0.5*pow(10, -6)
    pixeltom = 6*pow(10, -6)
    distance = 10.0
    propagation_type = 'Fraunhofer'
    k = wavenumber(wavelength)
    sample_field = np.zeros((150, 150), dtype=np.complex64)
    sample_field = np.zeros((150, 150), dtype=np.complex64)
    sample_field[
        65:85,
        65:85
    ] = 1
    sample_field = add_random_phase(sample_field)
    hologram = propagate_beam(
        sample_field,
        k,
        distance,
        pixeltom,
        wavelength,
        propagation_type
    )
    if propagation_type == 'Fraunhofer':
        # Uncomment if you want to match the physical size of hologram and input field.
        #from odak.wave import fraunhofer_equal_size_adjust
        #hologram         = fraunhofer_equal_size_adjust(hologram,distance,pixeltom,wavelength)
        propagation_type = 'Fraunhofer Inverse'
        distance = np.abs(distance)
    reconstruction = propagate_beam(
        hologram,
        k,
        -distance,
        pixeltom,
        wavelength,
        propagation_type
    )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
