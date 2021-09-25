# odak.learn.wave.gerchberg_saxton

`gerchberg_saxton(field,n_iterations,distance,dx,wavelength,slm_range=6.28,propagation_type='IR Fresnel')`

Definition to compute a hologram using an iterative method called Gerchberg-Saxton phase retrieval algorithm. 
For more on the method, see: `Gerchberg, Ralph W.`, `A practical algorithm for the determination of phase from image and diffraction plane pictures.` Optik 35 (1972): 237-246.

**Parameters:**

    field            : torch.cfloat
                       Complex field (MxN).
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    wavelength       : float
                       Wavelength of the electric field.
    slm_range        : float
                       Typically this is equal to two pi. See odak.wave.adjust_phase_only_slm_range() for more.
    propagation_type : str
                       Type of the propagation (IR Fresnel, TR Fresnel, Fraunhofer).

                       
**Returns**

    hologram         : torch.cfloat
                       Calculated complex hologram.
    reconstruction   : torch.cfloat
                       Calculated reconstruction using calculated hologram.

## Notes

To optimize a phase-only hologram using Gerchberg-Saxton algorithm, please follow and observe the below example:

```
import torch
from odak.learn.wave import gerchberg_saxton
from odak import np
wavelength              = 0.000000532
dx                      = 0.0000064
distance                = 0.2
target_field            = torch.zeros((500,500),dtype=torch.complex64)
target_field[0::50,:]  += 1
iteration_number        = 3
hologram,reconstructed  = gerchberg_saxton(
                                           target_field,
                                           iteration_number,
                                           distance,
                                           dx,
                                           wavelength,
                                           np.pi*2,
                                           'TR Fresnel'
                                          )
```



## See also

* [`Computer Generated-Holography`](../../../cgh.md)
* [`odak.learn.wave.stochastic_gradient_descent`](stochastic_gradient_descent.md)
* [`odak.learn.wave.point_wise`](point_wise.md)
