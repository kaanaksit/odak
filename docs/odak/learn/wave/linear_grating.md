# odak.learn.wave.linear_grating

`linear_grating(nx,ny,every=2,add=3.14,axis='x')`

A definition to generate a linear grating.

 
**Parameters:**

    nx         : int
                 Size of the output along X.
    ny         : int
                 Size of the output along Y.
    every      : int
                 Add the add value at every given number.
    add        : float
                 Angle to be added.
    axis       : string
                 Axis eiter X,Y or both.
                       
**Returns**

    field      : torch.tensor
                 Linear grating term.

## Notes

Here is a short example on how to use this function:

```
from odak.learn.wave import wavenumber,quadratic_phase_function
wavelength                 = 0.5*pow(10,-6)
pixeltom                   = 6*pow(10,-6)
distance                   = 10.0
resolution                 = [1080,1920]
k                          = wavenumber(wavelength)
plane_field                = linear_grating(
                                            resolution[0],
                                            resolution[1],
                                            every=2,
                                            add=3.14,
                                            axis='x'
                                           )
```

## See also

* [`Computer Generated-Holography`](../../../cgh.md)
