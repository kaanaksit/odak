# odak.learn.wave.custom

`custom(field,kernel)`

A definition to calculate convolution based Fresnel approximation for beam propagation. Curious reader can consult `Learned Holographic Light Transport`, Applied Optics (2021) by `Koray Kavaklı, Hakan Urey and Kaan Akşit`.

**Parameters:**

    field            : torch.complex
                       Complex field (MxN).
    kernel           : torch.complex
                       Custom complex kernel for beam propagation.
                       
**Returns**

    result           : torch.complex
                       Final complex field (MxN).

## Notes

Unless you know what you are doing, we do not suggest you to use this function directly. Rather stick to [`odak.learn.wave.propagate_beam`](propagate_beam.md) for  your beam propagation code. Note that this function can also be used as convolution operation between two complex arrays.

## See also

* [`Computer Generated-Holography`](../../../cgh.md)
* [`odak.learn.wave.propagate_beam`](propagate_beam.md)
