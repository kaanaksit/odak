# odak.learn.wave.quadratic_phase_function

::: odak.learn.wave.quadratic_phase_function
    selection:
        docstring_style: numpy

## Notes

Here is a short example on how to use this function:

```
from odak.learn.wave import wavenumber,quadratic_phase_function
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
