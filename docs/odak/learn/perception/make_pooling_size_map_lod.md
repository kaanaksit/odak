# odak.learn.perception.make_pooling_size_map_lod

`make_pooling_size_map_lod()`

This function is similar to make_pooling_size_map_pixels, but instead returns a map of LOD levels to sample from
to achieve the correct pooling region areas.

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

    pooling_size_map        : torch.tensor
                                The computed pooling size map, of size WxH.

## See also

[`Visual perception`](../../../perception.md)
