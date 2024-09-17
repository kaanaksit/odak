import sys
import torch
import odak
from odak.learn.wave import gerchberg_saxton


def test():
    wavelength = 532e-9
    dx = 6.4e-6
    distance = 0.2
    target_field = torch.zeros((500, 500), dtype=torch.complex64)
    target_field[0::50, :] += 1
    iteration_number = 3
    hologram, reconstructed = gerchberg_saxton(
                                               target_field,
                                               iteration_number,
                                               distance,
                                               dx,
                                               wavelength,
                                               2 * odak.pi,
                                               'Transfer Function Fresnel'
                                              )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
