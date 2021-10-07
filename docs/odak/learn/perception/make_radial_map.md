# odak.learn.perception.make_radial_map

`make_radial_map()`

Makes a simple radial map where each pixel contains distance in pixels from the chosen gaze location.

**Parameters**

    size    : tuple of ints
                Dimensions of the image
    gaze    : tuple of floats
                User's gaze (fixation point) in the image. Should be given as a tuple with normalized
                image coordinates (ranging from 0 to 1)

**Returns**
    radial_map  : torch.tensor
                    The radial map.

## See also

[`Visual perception`](../../../perception.md)
