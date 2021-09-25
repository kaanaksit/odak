# odak.learn.wave.point_wise

`point_wise(target,wavelength,distance,dx,device,lens_size=401)`

Naive point-wise hologram calculation method. 
For more information, refer to `Maimone, Andrew, Andreas Georgiou, and Joel S. Kollin`. `Holographic near-eye displays for virtual and augmented reality.` ACM Transactions on Graphics (TOG) 36.4 (2017): 1-16.

**Parameters:**

    target           : torch.float
                       float input target to be converted into a hologram (Target should be in range of 0 and 1).
    wavelength       : float
                       Wavelength of the electric field.
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    device           : torch.device
                       Device type (cuda or cpu)`.
    lens_size        : int
                       Size of lens for masking sub holograms(in pixels).

                       
**Returns**

    hologram         : torch.cfloat
                       Calculated complex hologram.

## Notes

To optimize a phase-only hologram using point wise algorithm, please follow and observe the below example:

```
import torch
from odak.learn.wave import point_wise
wavelength               = 0.000000515
dx                       = 0.000008
distance                 = 0.15
resolution               = [1080,1920]
target                   = torch.zeros(resolution[0],resolution[1])
target[540:600,960:1020] = 1
hologram                 = point_wise(
                                      target,
                                      wavelength,
                                      distance,
                                      dx,
                                      device,
                                      lens_size=401
                                     )
```

## See also

* [`Computer Generated-Holography`](../../../cgh.md)
* [`odak.learn.wave.stochastic_gradient_descent`](stochastic_gradient_descent.md)
* [`odak.learn.wave.gerchberg_saxton`](gerchberg_saxton.md)
