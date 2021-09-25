# odak.learn.wave.calculate_phase

`calculate_phase(field,deg=False)`

 Definition to calculate phase of a single or multiple given electric field(s).
 
**Parameters:**

    field        : torch.cfloat
                   Electric fields or an electric field.
    deg          : bool
                   If set True, the angles will be returned in degrees.
                       
**Returns**

    phase        : torch.float
                   Phase or phases of electric field(s) in radians.

## Notes

Regarding usage of this definition, you can find use cases in the engineering notes, specifically at [`Optimizing holograms using Odak`](../../../notes/hologram_optimization.md).

## See also

* [`Computer Generated-Holography`](../../../cgh.md)
