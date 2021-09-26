# odak.wave.adaptive_sampling_angular_spectrum

`adaptive_sampling_angular_spectrum(field,k,distance,dx,wavelength)`

A definition to calculate adaptive sampling angular spectrum based beam propagation. 
For more `Zhang, Wenhui, Hao Zhang, and Guofan Jin. "Adaptive-sampling angular spectrum method with full utilization of space-bandwidth product." Optics Letters 45.16 (2020): 4416-4419`.
 
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
