# odak.wave.rayleigh_sommerfeld

`rayleigh_sommerfeld(field,k,distance,dx,wavelength)`

Definition to compute beam propagation using Rayleigh-Sommerfeld's diffraction formula (Huygens-Fresnel Principle). 
For more see Section 3.5.2 in `Goodman, Joseph W. Introduction to Fourier optics. Roberts and Company Publishers, 2005`.
 
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
                       
**Returns**

    result           : np.complex
                       Final complex field (MxN).

## See also

* [`Computer Generated-Holography`](../../cgh.md)
