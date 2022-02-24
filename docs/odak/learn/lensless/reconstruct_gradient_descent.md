# odak.learn.lensless.reconstruct_gradient_descent

::: odak.learn.lensless.reconstruct_gradient_descent
    selection:
        docstring_style: numpy

## Notes

To reconstruct an image from a lensless measurement using gradient descent, please follow the below example:

```
from odak.learn.lensless import PhaseMaskCamera, reconstruct_gradient_descent

camera = PhaseMaskCamera(psf)
reconstruction = reconstruct_gradient_descent(camera, measurement, iters=10000, tol=1e-6)
```
