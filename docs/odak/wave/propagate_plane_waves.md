# odak.wave.propagate_plane_waves

`propagate_plane_waves(field,opd,k,w=0,t=0)`

Definition to propagate a field representing a plane wave at a particular distance and time.
 
**Parameters:**

    field        : complex
                   Complex field.
    opd          : float
                   Optical path difference in mm.
    k            : float
                   Wave number of a wave, see odak.wave.parameters.wavenumber for more.
    w            : float
                   Rotation speed of a wave, see odak.wave.parameters.rotationspeed for more.
    t            : float
                   Time in seconds.

                       
**Returns**

    new_field     : complex
                    A complex number that provides the resultant field in the complex form A*e^(j(wt+phi)).

## See also

* [`Computer Generated-Holography`](../../cgh.md)
