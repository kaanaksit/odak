# odak.wave.electric_field_per_plane_wave

`electric_field_per_plane_wave(amplitude,opd,k,phase=0,w=0,t=0)`

Definition to return state of a plane wave at a particular distance and time.
 
**Parameters:**

    amplitude    : float
                   Amplitude of a wave.
    opd          : float
                   Optical path difference in mm.
    k            : float
                   Wave number of a wave, see odak.wave.parameters.wavenumber for more.
    phase        : float
                   Initial phase of a wave.
    w            : float
                   Rotation speed of a wave, see odak.wave.parameters.rotationspeed for more.
    t            : float
                   Time in seconds.
                       
**Returns**

    field        : complex
                   A complex number that provides the resultant field in the complex form A*e^(j(wt+phi)).

## See also

* [`Computer Generated-Holography`](../../cgh.md)
