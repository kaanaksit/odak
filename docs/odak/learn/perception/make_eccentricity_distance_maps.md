# odak.learn.perception.make_eccentricity_distance_maps

`make_eccentricity_distance_maps()`

Makes a map of the eccentricity of each pixel in an image for a given fixation point, when displayed to a user on a flat screen. Assumes the viewpoint is located at the centre of the image, and the screen is  perpendicular to the viewing direction. Output in radians.

**Parameters**

    gaze_location           : tuple of floats
                                User's gaze (fixation point) in the image. Should be given as a tuple with normalized
                                image coordinates (ranging from 0 to 1)
    image_pixel_size        : tuple of ints
                                The size of the image in pixels, as a tuple of form (height, width)
    real_image_width        : float
                                The real width of the image as displayed. Units not important, as long as they
                                are the same as those used for real_viewing_distance
    real_viewing_distance   : float
                                The real distance from the user's viewpoint to the screen.

**Returns**

    eccentricity_map        : torch.tensor
                                The computed eccentricity map, of size WxH.
    distance_map            : torch.tensor
                                The computed distance map, of size WxH

## See also

[`Visual perception`](../../../perception.md)
