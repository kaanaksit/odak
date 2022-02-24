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
        self.h = rfft2(zero_pad(psf))

    def autocorrelation(self):
        return self.h * self.h.conj()

    def forward(self, scene):
        return crop_center(irfft2(self.h * rfft2(scene)))

    def adjoint(self, error):
        return irfft2(self.h.conj() * rfft2(zero_pad(error)))
