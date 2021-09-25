# odak.learn.wave.band_limited_angular_spectrum

`band_limited_angular_spectrum(field,k,distance,dx,wavelength)`

A definition to calculate bandlimited angular spectrum based beam propagation. 
For more `Matsushima, Kyoji, and Tomoyoshi Shimobaba`, `Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields.` Optics express 17.22 (2009): 19662-19673.

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

                       
**Returns**

    result           : torch.complex
                       Final complex field (MxN).

## Notes

Unless you know what you are doing, we do not suggest you to use this function directly. Rather stick to [`odak.learn.wave.propagate_beam`](propagate_beam.md) for  your beam propagation code. 

## See also

* [`Computer Generated-Holography`](../../../cgh.md)
* [`odak.learn.wave.propagate_beam`](propagate_beam.md)
