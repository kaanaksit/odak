import sys


def test():
    import torch
    from odak.learn.wave import point_wise
    wavelength = 0.000000515
    dx = 0.000008
    distance = 0.15
    resolution = [1080, 1920]
    device = torch.device("cpu")
    target = torch.zeros(resolution[0], resolution[1]).to(device)
    target[540:600, 960:1020] = 1
    hologram = point_wise(
        target,
        wavelength,
        distance,
        dx,
        device,
        lens_size=401
    )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
