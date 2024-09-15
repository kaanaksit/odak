import sys
import torch
from odak.learn.wave import point_wise


def test():
    wavelength = 515e-9
    dx = 8e-6
    distance = 0.15
    resolution = [1080, 1920]
    device = torch.device("cpu")
    target = torch.zeros(resolution[0], resolution[1], device = device)
    target[540:600, 960:1020] = 1
    hologram = point_wise(
                          target,
                          wavelength,
                          distance,
                          dx,
                          device,
                          lens_size = 401
                         )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
