# odak.wave.gerchberg_saxton

`gerchberg_saxton(field,n_iterations,distance,dx,wavelength,slm_range=6.28,propagation_type='IR Fresnel',initial_phase=None)`

Definition to compute a hologram using an iterative method called Gerchberg-Saxton phase retrieval algorithm. 
For more on the method, see: `Gerchberg, Ralph W. "A practical algorithm for the determination of phase from image and diffraction plane pictures." Optik 35 (1972): 237-246.`
 
**Parameters:**

    field            : np.complex64
                       Complex field (MxN).
    distance         : float
                       Propagation distance.
    dx               : float
                       Size of one single pixel in the field grid (in meters).
    wavelength       : float
                       Wavelength of the electric field.
    slm_range        : float
                       Typically this is equal to two pi. See odak.wave.adjust_phase_only_slm_range() for more.
    propagation_type : str
                       Type of the propagation (IR Fresnel, TR Fresnel, Fraunhofer).
    initial_phase    : np.complex64
                       Phase to be added to the initial value.
                       
**Returns**

    hologram         : np.complex
                       Calculated complex hologram.
    reconstruction   : np.complex
                       Calculated reconstruction using calculated hologram.

## See also

* [`Computer Generated-Holography`](../../cgh.md)
