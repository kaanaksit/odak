# Blur Loss

`odak.learn.perception.BlurLoss()`

`BlurLoss` implements two different blur losses. When `blur_source` is set to `False`, it implements blur_match, trying to match the input image to the blurred target image. This tries to match the source input image to a blurred version of the target.

When `blur_source` is set to `True`, it implements blur_lowpass, matching the blurred version of the input image to the blurred target image. This tries to match only the low frequencies of the source input image to the low frequencies of the target.

The interface is similar to other `pytorch` loss functions, but note that the gaze location must be provided in addition to the source and target images.

**Parameters:**

    alpha                   : float
                                parameter controlling foveation - larger values mean bigger pooling regions.
    real_image_width        : float 
                                The real width of the image as displayed to the user.
                                Units don't matter as long as they are the same as for real_viewing_distance.
    real_viewing_distance   : float 
                                The real distance of the observer's eyes to the image plane.
                                Units don't matter as long as they are the same as for real_image_width.
    n_pyramid_levels        : int 
                                Number of levels of the steerable pyramid. Note that the image is padded
                                so that both height and width are multiples of 2^(n_pyramid_levels), so setting this value
                                too high will slow down the calculation a lot.
    mode                    : str 
                                Foveation mode, either "quadratic" or "linear". Controls how pooling regions grow
                                as you move away from the fovea. We got best results with "quadratic".
    blur_source             : bool
                                If true, blurs the source image as well as the target before computing the loss.s

## Methods

`__call__()`

Calculates the Metameric Loss.

**Parameters**

        image               : torch.tensor
                                Image to compute loss for. Should be an RGB image in NCHW format (4 dimensions)
        target              : torch.tensor
                                Ground truth target image to compute loss for. Should be an RGB image in NCHW format (4 dimensions)
        gaze                : list
                                Gaze location in the image, in normalized image coordinates (range [0, 1]) relative to the top left of the image.
        
**Returns**

        loss                : torch.tensor
                                The computed loss.

## See also

[`Perception`](../../../perception.md)
