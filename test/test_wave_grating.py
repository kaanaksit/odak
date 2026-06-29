import sys
import numpy as np
from odak.wave import wavenumber, prism_phase_function, quadratic_phase_function


def test():
    wavelength = 532e-9
    pixeltom = 8e-6
    k = wavenumber(wavelength)
    sample_field = np.random.rand(150, 150).astype(np.complex64)
    lens_field = quadratic_phase_function(
        sample_field.shape[0], sample_field.shape[1], k, focal=0.3, dx=pixeltom
    )
    prism_phase_function(
        sample_field.shape[0],
        sample_field.shape[1],
        k,
        angle=0.1,
        dx=pixeltom,
        axis="x",
    )
    sample_field = sample_field * lens_field
    assert True


if __name__ == "__main__":
    sys.exit(test())
