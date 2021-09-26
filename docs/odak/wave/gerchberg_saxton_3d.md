# odak.wave.gerchberg_saxton_3d

`gerchberg_saxton_3d(fields,n_iterations,distances,dx,wavelength,slm_range=6.28,propagation_type='IR Fresnel',initial_phase=None,target_type='no constraint',coefficients=None)`

Definition to compute a multi plane hologram using an iterative method called Gerchberg-Saxton phase retrieval algorithm. 
For more on the method, see: `Zhou, Pengcheng, et al. "30.4: Multi‐plane holographic display with a uniform 3D Gerchberg‐Saxton algorithm." SID Symposium Digest of Technical Papers. Vol. 46. No. 1. 2015.`
 
**Parameters:**

    fields           : np.complex64
                       Complex fields (MxN).
    distances        : list
                       Propagation distances.
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
    target_type      : str
                       Target type. `No constraint` targets the input target as is. `Double constraint` follows the idea in this paper, which claims to suppress speckle: Chang, Chenliang, et al. "Speckle-suppressed phase-only holographic three-dimensional display based on double-constraint Gerchberg–Saxton algorithm." Applied optics 54.23 (2015): 6994-7001. 
                       
**Returns**

    hologram         : np.complex
                       Calculated complex hologram.

## See also

* [`Computer Generated-Holography`](../../cgh.md)
