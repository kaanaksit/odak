import sys


def test():
    import torch
    from odak.learn.tools import crop_center, zero_pad
    from odak.learn.lensless import PhaseMaskCamera, reconstruct_gradient_descent

    psf = torch.zeros(3, 256, 256)
    psf[:, 128, 128] = 1
    camera = PhaseMaskCamera(psf)

    scene = torch.zeros(3, 512, 512)
    measurement = camera.forward(scene)
    reconstruction = reconstruct_gradient_descent(camera, measurement, iters=1, tol=1e-6, disable_tqdm=True)
    torch.testing.assert_close(reconstruction, crop_center(scene))


if __name__ == '__main__':
    sys.exit(test())
