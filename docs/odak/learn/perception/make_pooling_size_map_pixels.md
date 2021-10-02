# odak.learn.perception.make_pooling_size_map_pizels

`make_pooling_size_map_pizels()`

Makes a map of the pooling size associated with each pixel in an image for a given fixation point, when displayed to a user on a flat screen. Follows the idea that pooling size (in radians) should be directly proportional to eccentricity (also in radians). 

Assumes the viewpoint is located at the centre of the image, and the screen is perpendicular to the viewing direction. Output is the width of the pooling region in pixels.

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
