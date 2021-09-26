# odak.wave.rayleigh_resolution

`rayleigh_resolution(diameter,focal=None,wavelength=0.0005)`

Definition for calculating the wavenumber of a plane wave
 
**Parameters:**

    diameter    : float
                  Diameter of a lens.
    focal       : float
                  Focal length of a lens, when focal length is provided, spatial resolution is provided at the focal plane. When focal length isn't provided angular resolution is provided.
    wavelength  : float
                  Wavelength of light.


                       
**Returns**

    resolution  : float
                  Resolvable angular or spatial spot size, see focal in parameters to know what to expect.
                  
## See also

* [`Computer Generated-Holography`](../../cgh.md)
