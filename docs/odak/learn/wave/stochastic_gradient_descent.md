# odak.learn.wave.stochastic_gradient_descent

::: odak.learn.wave.stochastic_gradient_descent
    selection:
        docstring_style: numpy

## Notes

To optimize a phase-only hologram using Gerchberg-Saxton algorithm, please follow and observe the below example:

```
import torch
from odak.learn.wave import stochastic_gradient_descent
wavelength               = 0.000000532
dx                       = 0.0000064
distance                 = 0.1
cuda                     = False
resolution               = [1080,1920]
target_field             = torch.zeros(resolution[0],resolution[1])
target_field[500::600,:] = 1
iteration_number         = 5
hologram,reconstructed   = stochastic_gradient_descent(
                                                       target_field,
                                                       wavelength,
                                                       distance,
                                                       dx,
                                                       resolution,
                                                       'TR Fresnel',
                                                       iteration_number,
                                                       learning_rate=0.1,
                                                       cuda=cuda
                                                      )
```

## See also

* [`Computer Generated-Holography`](../../../cgh.md)
* [`odak.learn.wave.gerchberg_saxton`](gerchberg_saxton.md)
* [`odak.learn.wave.point_wise`](point_wise.md)
