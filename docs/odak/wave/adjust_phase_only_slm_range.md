# odak.wave.adjust_phase_only_slm_range

`adjust_phase_only_slm_range(native_range,working_wavelength,native_wavelength)`

Definition for calculating the phase range of the Spatial Light Modulator (SLM) for a given wavelength. Here you prove maximum angle as the lower bound is typically zero. If the lower bound isn't zero in angles, you can use this very same definition for calculating lower angular bound as well.
 
**Parameters:**

    native_range       : float
                         Native range of the phase only SLM in radians (i.e. two pi).
    working_wavelength : float
                         Wavelength of the illumination source or some working wavelength.
    native_wavelength  : float
                         Wavelength which the SLM is designed for.
                       
**Returns**

    new_range          : float
                         Calculated phase range in radians.

## See also

* [`Computer Generated-Holography`](../../cgh.md)
