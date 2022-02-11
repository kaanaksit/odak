import torch.fft as fft

from ..tools import crop_center, zero_pad


class LenslessCamera:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class PhaseMaskCamera(LenslessCamera):
    """
    Forward and adjoint models for a phase-mask based lensless camera with a large, shift invariant kernel.
    The fourier transformed point spread function (PSF) is cached for efficiency.

    Parameters
    ----------
    psf         : torch.tensor
                  Real-valued point spread function measurement.
    """
    def __init__(self, psf):
        self.h = rft(zero_pad(psf))

    def autocorrelation(self):
        return self.h * self.h.conj()

    def forward(self, scene):
        return crop_center(irft(self.h * rft(scene)))

    def adjoint(self, error):
        return irft(self.h.conj() * rft(zero_pad(error)))


def rft(x):
    return fft.rfft2(fft.ifftshift(x))


def irft(x):
    return fft.fftshift(fft.irfft2(x))
