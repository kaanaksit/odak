import sys


def test():
    import torch
    import numpy as np
    from odak.learn.wave import gerchberg_saxton
    wavelength = 0.000000532
    dx = 0.0000064
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
        np.pi*2,
        'TR Fresnel'
    )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
