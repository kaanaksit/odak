# odak.wave.band_extended_angular_spectrum

`band_extended_angular_spectrum(field,k,distance,dx,wavelength)`

A definition to calculate bandextended angular spectrum based beam propagation. 
For more `Zhang, Wenhui, Hao Zhang, and Guofan Jin. "Band-extended angular spectrum method for accurate diffraction calculation in a wide propagation range." Optics Letters 45.6 (2020): 1543-1546`.
 
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
