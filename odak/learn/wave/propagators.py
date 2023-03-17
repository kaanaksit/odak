import torch
import numpy as np
from .util import wavenumber, generate_complex_field
from ..tools import zero_pad, crop_center, circular_binary_mask


class back_and_forth_propagator():
    """
    A light propagation model that propagates light to desired image plane with two separate propagations. We use this class in our various works including `Kavaklı et al., Realistic Defocus Blur for Multiplane Computer-Generated Holography`.
    """
    def __init__(
                 self,
                 wavelength = 515e-9,
                 pixel_pitch = 8e-6,
                 pad = True,
                 device = None
                ):
        """
        Parameters
        ----------
        wavelength         : float
                             Wavelength of light in meters.
        pixel_pitch        : float
                             Pixel pitch in meters.
        pad                : bool
                             Zero pad input-crop output flag.
        device             : torch.device
                             Device to be used for computation. For more see torch.device().
        """
        self.device = device
        if isinstance(self.device, type(None)):
            self.device = torch.device('cpu')
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength
        self.wavenumber = wavenumber(self.wavelength)
        self.pad = pad
        self.distances = []
        self.kernels = []
 

    def generate_kernel(self, nu, nv, dx = 8e-6, wavelength = 515e-9, distance = 0.):
        """
        Internal function used for self.propagate().
        """
        x = dx * float(nu)
        y = dx * float(nv)
        fx = torch.linspace(
                            -1 / (2 * dx) + 0.5 / (2 * x),
                            1 / (2 * dx) - 0.5 / (2 * x),
                            nu,
                            dtype = torch.float32,
                            device = self.device
                           )
        fy = torch.linspace(
                            -1 / (2 * dx) + 0.5 / (2 * y),
                            1 / (2 * dx) - 0.5 / (2 * y),
                            nv,
                            dtype = torch.float32,
                            device = self.device
                           )
        FY, FX = torch.meshgrid(fx, fy, indexing='ij')
        HH_exp = 2 * np.pi * torch.sqrt(1 / wavelength ** 2 - (FX ** 2 + FY ** 2))
        distance = torch.tensor([distance], device = self.device)
        H_exp = torch.mul(HH_exp, distance)
        fx_max = 1 / torch.sqrt((2 * distance * (1 / x))**2 + 1) / wavelength
        fy_max = 1 / torch.sqrt((2 * distance * (1 / y))**2 + 1) / wavelength
        H_filter = ((torch.abs(FX) < fx_max) & (torch.abs(FY) < fy_max)).clone().detach()
        H = generate_complex_field(H_filter, H_exp)
        return H


    def propagate(self, field, H, zero_padding = [True, False, True]):
        """
        Internal function used in propagation. It is a copy of odak.learn.wave.band_limited_angular_spectrum().
        """
        if zero_padding[0] == True:
            field = zero_pad(field)
        U1 = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field)))
        if zero_padding[1] == True:
            H = zero_pad(H)
            U1 = zero_pad(U1)
        U2 = H * U1
        result = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(U2)))
        if zero_padding[2] == True:
            return crop_center(result)
        return result


    def __call__(self, field, distances):
        """
        Model used to generate a complex field in the vicinity of a spatial light modulator.

        Parameters
        ----------
        field           : torch.tensor
                          Input field [m x n].
        distances       : list
                          Distance to propagate.

        Returns
        -------
        final_field     : torch.tensor
                          Propagated final complex field [m x n].
        """
        if self.pad == True:
            m = 2
        else:
            m = 1
        if distances[0] not in self.distances:
            self.distances.append(distances[0])
            kernel_forward = self.generate_kernel(
                                                  field.shape[-2] * m,
                                                  field.shape[-1] * m,
                                                  self.pixel_pitch,
                                                  self.wavelength,
                                                  distances[0]
                                                 )
            self.kernels.append(kernel_forward)
        else:
            kernel_id = self.distances.index(distances[0])
            kernel_forward = self.kernels[kernel_id]
        if distances[1] not in self.distances:
            self.distances.append(distances[1])
            kernel_backward = self.generate_kernel(
                                                   field.shape[-2] * m,
                                                   field.shape[-1] * m,
                                                   self.pixel_pitch,
                                                   self.wavelength,
                                                   distances[1]
                                                  )
            self.kernels.append(kernel_backward)
        else:
            kernel_id = self.distances.index(distances[1])
            kernel_backward = self.kernels[kernel_id]
        kernel = kernel_forward * kernel_backward
        final_field = self.propagate(
                                     field,
                                     kernel,
                                     zero_padding = [self.pad, False, self.pad]
                                    )
        return final_field
    


class forward_propagator():
    """
    A light propagation model that propagates light to desired image plane with two separate propagations. We use this class in our various works including `Kavaklı et al., Realistic Defocus Blur for Multiplane Computer-Generated Holography`.
    """
    def __init__(
                 self,
                 wavelength = 515e-9,
                 pixel_pitch = 8e-6,
                 pad = True,
                 device = None
                ):
        """
        Parameters
        ----------
        wavelength         : float
                             Wavelength of light in meters.
        pixel_pitch        : float
                             Pixel pitch in meters.
        pad                : bool
                             Zero pad input-crop output flag.
        device             : torch.device
                             Device to be used for computation. For more see torch.device().
        """
        self.device = device
        if isinstance(self.device, type(None)):
            self.device = torch.device('cpu')
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength
        self.wavenumber = wavenumber(self.wavelength)
        self.pad = pad
        self.distances = []
        self.kernels = []
 

    def generate_kernel(self, nu, nv, dx = 8e-6, wavelength = 515e-9, distance = 0.):
        """
        Internal function used for self.propagate().
        """
        x = dx * float(nu)
        y = dx * float(nv)
        fx = torch.linspace(
                            -1 / (2 * dx) + 0.5 / (2 * x),
                            1 / (2 * dx) - 0.5 / (2 * x),
                            nu,
                            dtype = torch.float32,
                            device = self.device
                           )
        fy = torch.linspace(
                            -1 / (2 * dx) + 0.5 / (2 * y),
                            1 / (2 * dx) - 0.5 / (2 * y),
                            nv,
                            dtype = torch.float32,
                            device = self.device
                           )
        FY, FX = torch.meshgrid(fx, fy, indexing='ij')
        HH_exp = 2 * np.pi * torch.sqrt(1 / wavelength ** 2 - (FX ** 2 + FY ** 2))
        distance = torch.tensor([distance], device = self.device)
        H_exp = torch.mul(HH_exp, distance)
        fx_max = 1 / torch.sqrt((2 * distance * (1 / x))**2 + 1) / wavelength
        fy_max = 1 / torch.sqrt((2 * distance * (1 / y))**2 + 1) / wavelength
        aperture_size = int(3e-3 / dx)
        mask = circular_binary_mask(nu, nv, aperture_size).to(self.device) * 1.
        H_filter = ((torch.abs(FX) < fx_max) & (torch.abs(FY) < fy_max)).clone().detach() * mask
        H = generate_complex_field(H_filter, H_exp)
        return H


    def propagate(self, field, H, zero_padding = [True, False, True]):
        """
        Internal function used in propagation. It is a copy of odak.learn.wave.band_limited_angular_spectrum().
        """
        if zero_padding[0] == True:
            field = zero_pad(field)
        U1 = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field)))
        if zero_padding[1] == True:
            H = zero_pad(H)
            U1 = zero_pad(U1)
        U2 = H * U1
        result = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(U2)))
        if zero_padding[2] == True:
            return crop_center(result)
        return result


    def __call__(self, field, distance):
        """
        Model used to generate a complex field in the vicinity of a spatial light modulator.

        Parameters
        ----------
        field           : torch.tensor
                          Input field [m x n].
        distance        : float
                          Distance to propagate.

        Returns
        -------
        final_field     : torch.tensor
                          Propagated final complex field [m x n].
        """
        if self.pad == True:
            m = 2
        else:
            m = 1
        if distance not in self.distances:
            self.distances.append(distance)
            kernel_forward = self.generate_kernel(
                                                  field.shape[-2] * m,
                                                  field.shape[-1] * m,
                                                  self.pixel_pitch,
                                                  self.wavelength,
                                                  distance
                                                 )
            self.kernels.append(kernel_forward)
        else:
            kernel_id = self.distances.index(distance)
            kernel_forward = self.kernels[kernel_id]
        kernel = kernel_forward
        final_field = self.propagate(
                                     field,
                                     kernel,
                                     zero_padding = [self.pad, False, self.pad]
                                    )
        return final_field
