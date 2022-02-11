import torch
from tqdm import trange

from ..tools import crop_center, zero_pad 


def reconstruct_gradient_descent(camera, measurement, iters=10000, tol=1e-6, disable_tqdm=None):
    """
    Reconstruct an image from a lensless camera using vanilla, unregularized gradient descent.

    Parameters
    ----------
    camera      : LenslessCamera
                  Lensless camera forward and adjoint model.

    measurement : torch.tensor
                  Lensless measurement.
    """
    output = torch.zeros_like(zero_pad(measurement))
    alpha = 1.8 / torch.max(camera.autocorrelation().abs())

    prev_err = 1e6
    for i in trange(0, iters, disable=disable_tqdm):
        forward  = camera.forward(output)
        error    = forward - measurement
        if (error - prev_err).sum().abs() < tol:
            break
        prev_err = error
        gradient = camera.adjoint(error)
        output   = output - alpha * gradient
        output   = torch.maximum(output, torch.tensor(0))

    return crop_center(torch.maximum(output, torch.tensor(0)))
