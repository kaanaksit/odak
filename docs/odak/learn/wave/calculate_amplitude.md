# odak.learn.wave.calculate_amplitude

`calculate_amplitude(field)`

Definition to calculate amplitude of a single or multiple given electric field(s).
 
**Parameters:**

    field        : torch.cfloat
                   Electric fields or an electric field.
    deg          : bool
                   If set True, the angles will be returned in degrees.
                       
**Returns**

    amplitude    : torch.float
                   Amplitude or amplitudes of electric field(s).

## Notes

Regarding usage of this definition, you can find use cases in the engineering notes, specifically at [`Optimizing holograms using Odak`](../../../notes/hologram_optimization.md).

## See also

* [`Computer Generated-Holography`](../../../cgh.md)
