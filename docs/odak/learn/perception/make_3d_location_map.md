# odak.learn.perception.make_3d_location_map

`make_3d_location_map()`

Makes a map of the real 3D location that each pixel in an image corresponds to, when displayed to a user on a flat screen. Assumes the viewpoint is located at the centre of the image, and the screen is perpendicular to the viewing direction.

**Parameters**

    image_pixel_size        : tuple of ints 
                                The size of the image in pixels, as a tuple of form (height, width)
    real_image_width        : float
                                The real width of the image as displayed. Units not important, as long as they
                                are the same as those used for real_viewing_distance
    real_viewing_distance   : float 
                                The real distance from the user's viewpoint to the screen.
    
**Returns**

    map                     : torch.tensor
                                The computed 3D location map, of size 3xWxH.
