# odak.learn.wave.propagate_beam

::: odak.learn.wave.propagate_beam
    selection:
        docstring_style: numpy

## Notes
We provide a sample usage of this function as below.

```
from odak.learn.wave import propagate_beam,generate_complex_field,wavenumber
from odak.learn.tools import zero_pad
import torch

wavelength                 = 0.5*pow(10,-6)
pixeltom                   = 6*pow(10,-6)
distance                   = 0.2
propagation_type           = 'TR Fresnel'
k                          = wavenumber(wavelength)
sample_phase               = torch.rand((500,500))
sample_amplitude           = torch.zeros((500,500))
sample_amplitude[
                 240:260,
                 240:260
                ]          = 1000
sample_field               = generate_complex_field(sample_amplitude,sample_phase)


sample_field               = zero_pad(sample_field)
reconstruction             = propagate_beam(
                                            sample_field,
                                            k,
                                            distance,
                                            pixeltom,
                                            wavelength,
                                            propagation_type
                                           )
```

## See also

* [`Computer Generated-Holography`](../../../cgh.md)
