# odak.wave.prism_phase_function

`prism_phase_function(nx,ny,k,angle,dx=0.001,axis='x')`

A definition to generate 2D phase function that represents a prism. See Goodman's Introduction to Fourier Optics book for more.
 
**Parameters:**

    nx         : int
                 Size of the output along X.
    ny         : int
                 Size of the output along Y.
    k          : odak.wave.wavenumber
                 See odak.wave.wavenumber for more.
    angle      : float
                 Tilt angle of the prism in degrees.
    dx         : float
                 Pixel pitch.
    axis       : str
                 Axis of the prism.
                       
**Returns**

    prism      : ndarray
                 Generated phase function for a prism.
                 
## See also

* [`Computer Generated-Holography`](../../cgh.md)
