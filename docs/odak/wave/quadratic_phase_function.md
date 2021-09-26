# odak.wave.quadratic_phase_function

`quadratic_phase_function(nx,ny,k,focal=0.4,dx=0.001,offset=[0,0])`

A definition to generate 2D quadratic phase function, which is typically use to represent lenses.
 
**Parameters:**

    nx         : int
                 Size of the output along X.
    ny         : int
                 Size of the output along Y.
    k          : odak.wave.wavenumber
                 See odak.wave.wavenumber for more.
    focal      : float
                 Focal length of the quadratic phase function.
    dx         : float
                 Pixel pitch.
    offset     : list
                 Deviation from the center along X and Y axes.
                       
**Returns**

    function   : ndarray
                 Generated quadratic phase function.

## See also

* [`Computer Generated-Holography`](../../cgh.md)
