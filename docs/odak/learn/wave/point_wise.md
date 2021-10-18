# odak.learn.wave.point_wise

::: odak.learn.wave.point_wise
    selection:
        docstring_style: numpy

## Notes

To optimize a phase-only hologram using point wise algorithm, please follow and observe the below example:

```
import torch
from odak.learn.wave import point_wise
wavelength               = 0.000000515
dx                       = 0.000008
distance                 = 0.15
resolution               = [1080,1920]
cuda                     = True
device                   = torch.device("cuda" if cuda else "cpu")
target                   = torch.zeros(resolution[0],resolution[1]).to(device)
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
