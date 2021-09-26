# odak.wave.fraunhofer_equal_size_adjust

`fraunhofer_equal_size_adjust(field,distance,dx,wavelength)`

A definition to match the physical size of the original field with the propagated field.
 
**Parameters:**

    field            : np.complex
                       Complex field (MxN).
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    wavelength       : float
                       Wavelength of the electric field.
                       
**Returns**

    new_field        : np.complex
                       Final complex field (MxN).

## See also

* [`Computer Generated-Holography`](../../cgh.md)
