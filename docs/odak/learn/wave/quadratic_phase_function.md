# odak.learn.wave.quadratic_phase_function

`quadratic_phase_function(nx,ny,k,focal=0.4,dx=0.001,offset=[0,0])`

 A definition to generate 2D quadratic phase function, which is typically use to represent lenses.
 
**Parameters:**

    nx         : int
                 Size of the output along X.
    ny         : int
                 Size of the output along Y.
    k          : odak.wave.wavenumber
                 See odak.wave.wavenumber for more.
    focal      : float
                 Focal length of the quadratic phase function.
    dx         : float
                 Pixel pitch.
    offset     : list
                 Deviation from the center along X and Y axes.

                       
**Returns**

    function   : torch.tensor
                 Generated quadratic phase function.

## Notes

Here is a short example on how to use this function:

```
from odak.wave import wavenumber,quadratic_phase_function
wavelength                 = 0.5*pow(10,-6)
pixeltom                   = 6*pow(10,-6)
distance                   = 10.0
resolution                 = [1080,1920]
k                          = wavenumber(wavelength)

lens_field                 = quadratic_phase_function(
                                                      resolution[0],
                                                      resolution[1],
                                                      k,
                                                      focal=0.3,
                                                      dx=pixeltom
                                                     )
```

## See also

* [`Computer Generated-Holography`](../../../cgh.md)
