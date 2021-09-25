# odak.learn.wave.prism_phase_function

`prism_phase_function(nx,ny,k,angle,dx=0.001,axis='x')`

A definition to generate 2D phase function that represents a prism. 
See Goodman's Introduction to Fourier Optics book for more.

 
**Parameters:**

    nx         : int
                 Size of the output along X.
    ny         : int
                 Size of the output along Y.
    k          : odak.wave.wavenumber
                 See odak.wave.wavenumber for more.
    angle      : float
                 Tilt angle of the prism in degrees.
    dx         : float
                 Pixel pitch.
    axis       : str
                 Axis of the prism.

                       
**Returns**

    prism      : torch.tensor
                 Generated phase function for a prism.

## Notes

Here is a short example on how to use this function:

```
from odak.wave import wavenumber,quadratic_phase_function
wavelength                 = 0.5*pow(10,-6)
pixeltom                   = 6*pow(10,-6)
distance                   = 10.0
resolution                 = [1080,1920]
k                          = wavenumber(wavelength)

lens_field                 = prism_phase_function(
                                                  resolution[0],
                                                  resolution[1],
                                                  k,
                                                  fangle=0.1,
                                                  dx=pixeltom,
                                                  axis='x'
                                                 )
```

## See also

* [`Computer Generated-Holography`](../../../cgh.md)
