# odak.wave.fraunhofer

`fraunhofer(field,k,distance,dx,wavelength)`

A definition to calculate Fraunhofer based beam propagation.
 
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
