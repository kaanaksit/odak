# odak.learn.wave.propagate_beam

`odak.learn.propagate_beam(field,k,distance,dx,wavelength,propagation_type='IR Fresnel',kernel=None)`

Definitions for Fresnel impulse respone (IR), Fresnel Transfer Function (TF), Fraunhofer diffraction in accordence with "Computational Fourier Optics" by David Vuelz.

**Parameters:**

    field            : torch.complex
                       Complex field (MxN).
    k                : odak.wave.wavenumber
                       Wave number of a wave, see odak.wave.wavenumber for more.
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    wavelength       : float
                       Wavelength of the electric field.
    propagation_type : str
                       Type of the propagation (IR Fresnel, TR Fresnel, Fraunhofer).
    kernel           : torch.complex
                       Custom complex kernel.
**Returns**

    result           : torch.complex128
                       Final complex field (MxN).

## Notes
