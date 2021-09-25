# odak.learn.wave.set_amplitude

`set_amplitude(field,amplitude)`

Definition to keep phase as is and change the amplitude of a given field.
 
**Parameters:**

    field        : torch.cfloat
                   Complex field.
    amplitude    : torch.cfloat or torch.float
                   Amplitudes.

                       
**Returns**

    new_field    : torch.cfloat
                   Complex field.

## Notes

Regarding usage of this definition, you can find use cases in the engineering notes, specifically at [`Optimizing holograms using Odak`](../../../notes/optimizing_holograms_using_odak.md).

## See also

* [`Computer Generated-Holography`](../../../cgh.md)
