# Radially Varying Blur

`odak.learn.perception.RadiallyVaryingBlur()`

The `RadiallyVaryingBlur` class provides a way to apply a radially varying blur to an image. Given a gaze location and information about the image and foveation, it applies a blur that will achieve the proper pooling size. The pooling size is chosen to appear the same at a range of display sizes and viewing distances, for a given `alpha` parameter value. For more information on how the pooling sizes are computed, please see [link coming soon]().

The blur is accelerated by generating and sampling from MIP maps of the input image.

This class caches the foveation information. This means that if it is run repeatedly with the same foveation parameters, gaze location and image size (e.g. in an optimisation loop) it won't recalculate the pooling maps.

If you are repeatedly applying blur to images of different sizes (e.g. a pyramid) for best performance use one instance of this class per image size.

## Methods

`blur()`

Apply the radially varying blur to an image.

**Parameters**

        image                   : torch.tensor
                                    The image to blur, in NCHW format.
        alpha                   : float
                                    parameter controlling foveation - larger values mean bigger pooling regions.
        real_image_width        : float 
                                    The real width of the image as displayed to the user.
                                    Units don't matter as long as they are the same as for real_viewing_distance.
        real_viewing_distance   : float 
                                    The real distance of the observer's eyes to the image plane.
                                    Units don't matter as long as they are the same as for real_image_width.
        centre                  : tuple of floats
                                    The centre of the radially varying blur (the gaze location).
                                    Should be a tuple of floats containing normalised image coordinates in range [0,1]
        mode                    : str 
                                    Foveation mode, either "quadratic" or "linear". Controls how pooling regions grow
                                    as you move away from the fovea. We got best results with "quadratic".
        
**Returns**

        output                  : torch.tensor
                                    The blurred image

## See also

[`Perception`](../../../perception.md)
