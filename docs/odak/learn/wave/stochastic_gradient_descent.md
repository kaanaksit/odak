# odak.learn.wave.stochastic_gradient_descent

`stochastic_gradient_descent(field,wavelength,distance,dx,resolution,propogation_type,n_iteration=100,loss_function=None,cuda=False,learning_rate=0.1)`

Definition to generate phase and reconstruction from target image via stochastic gradient descent.
For more on the method, see: `Yuxin Chen, Yuejie Chi, Jianqing Fan, and Cong Ma.`, `Gradient descent with random initialization: fast global convergence for nonconvex phase retrieval.` Mathematical Programming 176 (2019), 1436–4646.
In addition this work is a great related read: `Peng, Y., Choi, S., Padmanaban, N., & Wetzstein, G.` (2020). `Neural holography with camera-in-the-loop training.` ACM Transactions on Graphics (TOG), 39(6), 1-14.
**Parameters:**

    field                   : torch.Tensor
                              Target field intensity.
    wavelength              : double
                              Set if the converted array requires gradient.
    distance                : double
                              Hologaram plane distance wrt SLM plane
    dx                      : float
                              SLM pixel pitch
    resolution              : array
                              SLM resolution
    propogation type        : str
                              Type of the propagation (IR Fresnel, Angular Spectrum, Bandlimited Angular Spectrum, TR Fresnel, Fraunhofer)
    n_iteration:            : int
                              Max iteratation 
    loss_function:          : function
                              If none it is set to be l2 loss
    cuda                    : boolean
                              GPU enabled
    learning_rate           : float
                              Learning rate.

                       
**Returns**

    hologram                : torch.Tensor
                              Phase only hologram as torch array

    reconstruction_intensity: torch.Tensor
                              Reconstruction as torch array

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
* [`odak.learn.wave.stochastic_gradient_descent`](stochastic_gradient_descent.md)
* [`odak.learn.wave.point_wise`](point_wise.md)
