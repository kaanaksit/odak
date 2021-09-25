# odak.learn.wave.produce_phase_only_slm_pattern

`produce_phase_only_slm_pattern(hologram,slm_range,bits=8,default_range=6.28,illumination=None)`

Definition for producing a pattern for a phase only Spatial Light Modulator (SLM) using a given field.
 
**Parameters:**

    hologram           : torch.cfloat
                         Input holographic field.
    slm_range          : float
                         Range of the phase only SLM in radians for a working wavelength (i.e. two pi). See odak.wave.adjust_phase_only_slm_range() for more.
    filename           : str
                         Optional variable, if provided the patterns will be save to given location.
    bits               : int
                         Quantization bits.
    default_ramge      : float
                         Default range of phase only SLM.
    illumination       : torch.tensor
                         Spatial illumination distribution.


                       
**Returns**

    pattern            : torch.cfloat
                         Adjusted phase only pattern.
    hologram_digital   : np.int
                         Digital representation of the hologram.

## Notes

Regarding usage of this definition, you can find use cases in the engineering notes, specifically at [`Optimizing holograms using Odak`](../../../notes/optimizing_holograms_using_odak.md).

## See also

* [`Computer Generated-Holography`](../../../cgh.md)
