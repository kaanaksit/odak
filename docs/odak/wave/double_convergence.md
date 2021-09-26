# odak.wave.double_convergence

`double_convergence(nx,ny,k,r,dx)`

A definition to generate initial phase for a Gerchberg-Saxton method. For more details consult Sun, Peng, et al. "Holographic near-eye display system based on double-convergence light Gerchberg-Saxton algorithm." Optics express 26.8 (2018): 10140-10151.
 
**Parameters:**

    nx         : int
                 Size of the output along X.
    ny         : int
                 Size of the output along Y.
    k          : odak.wave.wavenumber
                 See odak.wave.wavenumber for more.
    r          : float
                 The distance between location of a light source and an image plane.
    dx         : float
                 Pixel pitch.
                       
**Returns**

    function   : ndarray
                 Generated phase pattern for a Gerchberg-Saxton method.

## See also

* [`Computer Generated-Holography`](../../cgh.md)
