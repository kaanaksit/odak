# odak.learn.wave.propagate_beam

`odak.learn.propagate_beam(field,k,distance,dx,wavelength,propagation_type='IR Fresnel',kernel=None)`

This is the primary holographic light transport definition. It links to Fresnel impulse respone (IR), Fresnel Transfer Function (TF), Fraunhofer diffraction. Curious users can consult "Computational Fourier Optics" by David Vuelz.

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
    propagation_type : str
                       Type of the propagation (IR Fresnel, TR Fresnel, Fraunhofer).
    kernel           : torch.complex
                       Custom complex kernel.
**Returns**

    result           : torch.complex128
                       Final complex field (MxN).

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
hologram_torch             = propagate_beam(
                                            sample_field,
                                            k,
                                            distance,
                                            pixeltom,
                                            wavelength,
                                            propagation_type
                                           )
```

## See also

* [`Computer Generated-Holography`](../../cgh.md)
