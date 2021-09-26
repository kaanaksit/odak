# odak.wave.produce_phase_only_slm_pattern

`produce_phase_only_slm_pattern(hologram,slm_range,filename=None,bits=8,default_range=6.28,illumination=None)`

Definition for producing a pattern for a phase only Spatial Light Modulator (SLM) using a given field.
 
**Parameters:**

    hologram           : np.complex64
                         Input holographic field.
    slm_range          : float
                         Range of the phase only SLM in radians for a working wavelength (i.e. two pi). See odak.wave.adjust_phase_only_slm_range() for more.
    filename           : str
                         Optional variable, if provided the patterns will be save to given location.
    bits               : int
                         Quantization bits.
    default_range      : float 
                         Default range of phase only SLM.
    illumination       : np.ndarray
                         Spatial illumination distribution.
                       
**Returns**

    pattern            : np.complex64
                         Adjusted phase only pattern.
    hologram_digital   : np.int
                         Digital representation of the hologram.

## See also

* [`Computer Generated-Holography`](../../cgh.md)
