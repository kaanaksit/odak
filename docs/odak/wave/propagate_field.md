# odak.wave.propagate_field

`propagate_field(points0,points1,field0,wave_number,direction=1)`

Definition to propagate a field from points to an another points in space: propagate a given array of spherical sources to given set of points in space.
 
**Parameters:**

    points0       : ndarray
                    Start points (i.e. odak.tools.grid_sample).
    points1       : ndarray
                    End points (ie. odak.tools.grid_sample).
    field0        : ndarray
                    Field for given starting points.
    wave_number   : float
                    Wave number of a wave, see odak.wave.wavenumber for more.
    direction     : float
                    For propagating in forward direction set as 1, otherwise -1.
                       
**Returns**

    field1        : ndarray
                    Field for given end points.

## See also

* [`Computer Generated-Holography`](../../cgh.md)
