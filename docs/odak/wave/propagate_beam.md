# odak.wave.propagate_beam

`propagate_beam(field,k,distance,dx,wavelength,propagation_type='IR Fresnel')`

Definitions for Fresnel Impulse Response (IR), Angular Spectrum (AS), Bandlimited Angular Spectrum (BAS), Fresnel Transfer Function (TF), Fraunhofer diffraction in accordence with `Computational Fourier Optics` by David Vuelz. 
For more on Bandlimited Fresnel impulse response also known as Bandlimited Angular Spectrum method see `Band-limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields`.
 
**Parameters:**

    field            : np.complex
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
                       Type of the propagation (IR Fresnel, Angular Spectrum, Bandlimited Angular Spectrum, TR Fresnel, Fraunhofer).
                       
**Returns**

    result           : np.complex
                       Final complex field (MxN).

## See also

* [`Computer Generated-Holography`](../../cgh.md)
