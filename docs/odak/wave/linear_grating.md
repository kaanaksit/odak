# odak.wave.linear_grating

`linear_grating(nx,ny,every=2,add=3.14,axis='x')`

A definition to generate a linear grating.
 
**Parameters:**

    nx         : int
                 Size of the output along X.
    ny         : int
                 Size of the output along Y.
    every      : int
                 Add the add value at every given number.
    add        : float
                 Angle to be added.
    axis       : string
                 Axis eiter X,Y or both.
                       
**Returns**

    field      : ndarray
                 Linear grating term.

## See also

* [`Computer Generated-Holography`](../../cgh.md)
