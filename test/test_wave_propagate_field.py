from odak import tools
from odak import wave
import numpy as np
import sys


def test():
    no0 = [100, 100]
    no1 = [100, 100]
    size0 = [10., 10.]
    size1 = [10., 10.]
    wavelength = 0.005
    surface_location_0 = [0., 0., 0.]
    surface_location_1 = [0., 0., 10.]
    wave_number = wave.wavenumber(wavelength)
    samples_surface_0 = tools.grid_sample(
        no=no0, size=size0, center=surface_location_0)
    samples_surface_1 = tools.grid_sample(
        no=no1, size=size1, center=surface_location_1)
    field_0 = np.zeros((no0[0], no0[1]))
    field_0[50, 50] = 1.
    field_0[0, 20] = 2.
    field_0 = field_0.reshape((no0[0]*no0[1]))
    # Propagates to a specific plane.
    field_1 = wave.propagate_field(
        samples_surface_0,
        samples_surface_1,
        field_0,
        wave_number,
        direction=1
    )
    # Reconstruction: propagates back from that specific plane to start plane.
    field_2 = wave.propagate_field(
        samples_surface_1,
        samples_surface_0,
        field_1,
        wave_number,
        direction=-1
    )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
