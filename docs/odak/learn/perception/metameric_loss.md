# Metameric Loss

`odak.learn.perception.MetamericLoss()`

The `MetamericLoss` class provides a perceptual loss function.

Rather than exactly match the source image to the target, it tries to ensure the source is a *metamer* to the target image.

Its interface is similar to other `pytorch` loss functions, but note that the gaze location must be provided in addition to the source and target images.

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
    n_orientations          : int 
                                Number of orientations in the steerable pyramid. Can be 1, 2, 4 or 6.
                                Increasing this will increase runtime.
    use_l2_foveal_loss      : bool 
                                If true, for all the pixels that have pooling size 1 pixel in the 
                                largest scale will use direct L2 against target rather than pooling over pyramid levels.
                                In practice this gives better results when the loss is used for holography.
    fovea_weight            : float 
                                A weight to apply to the foveal region if use_l2_foveal_loss is set to True.
    use_radial_weight       : bool 
                                If True, will apply a radial weighting when calculating the difference between
                                the source and target stats maps. This weights stats closer to the fovea more than those
                                further away.
    use_fullres_l0          : bool 
                                If true, stats for the lowpass residual are replaced with blurred versions
                                of the full-resolution source and target images

## Methods

`__call__()`

Calculates the Metameric Loss.

**Parameters**

        image               : torch.tensor
                                Image to compute loss for. Should be an RGB image in NCHW format (4 dimensions)
        target              : torch.tensor
                                Ground truth target image to compute loss for. Should be an RGB image in NCHW format (4 dimensions)
        image_colorspace    : str
                                The current colorspace of your image and target. Ignored if input does not have 3 channels.
                                accepted values: RGB, YCrCb.
        gaze                : list
                                Gaze location in the image, in normalized image coordinates (range [0, 1]) relative to the top left of the image.
        visualise_loss      : bool
                                Shows a heatmap indicating which parts of the image contributed most to the loss. 
        
**Returns**

        loss                : torch.tensor
                                The computed loss.

## See also

[`Perception`](../../../perception.md)
