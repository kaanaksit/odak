# odak.learn.wave.impulse_response_fresnel

`impulse_response_fresnel(field,k,distance,dx,wavelength)`

A definition to calculate impulse response based Fresnel approximation for beam propagation.
Curious users can consult `Computational Fourier Optics` by David Vuelz.
**Refer to Issue 19 for more. This definition is unreliable.**

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

Unless you know what you are doing, we do not suggest you to use this function directly. 
Rather stick to [`odak.learn.wave.propagate_beam`](propagate_beam.md) for  your beam propagation code. 

## See also

* [`Computer Generated-Holography`](../../../cgh.md)
* [`odak.learn.wave.propagate_beam`](propagate_beam.md)
