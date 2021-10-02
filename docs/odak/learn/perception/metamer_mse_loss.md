# Metameric Loss

`odak.learn.perception.MetamerMSELoss()`

The `MetamerMSELoss` class provides a perceptual loss function. This generates a metamer for the target image, and then optimises the source image to be the same as this target image metamer.

Please note this is different to [`MetamericLoss`](metameric_loss.md) which optimises the source image to be any metamer of the target image.

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

## Methods

`__call__()`

Calculates the Metameric MSE Loss.

**Parameters**

        image   : torch.tensor
                Image to compute loss for. Should be an RGB image in NCHW format (4 dimensions)
        target  : torch.tensor
                Ground truth target image to compute loss for. Should be an RGB image in NCHW format (4 dimensions)
        gaze    : list
                Gaze location in the image, in normalized image coordinates (range [0, 1]) relative to the top left of the image.

**Returns**

        loss                : torch.tensor
                                The computed loss.

`gen_metamer(image, gaze)`

Generates a metamer for an image, following the method in [this paper](https://dl.acm.org/doi/abs/10.1145/3450626.3459943)
This function can be used on its own to generate a metamer for a desired image.

**Parameters**

        image   : torch.tensor
                Image to compute metamer for. Should be an RGB image in NCHW format (4 dimensions)
        gaze    : list
                Gaze location in the image, in normalized image coordinates (range [0, 1]) relative to the top left of the image.
        
**Returns**

        metamer : torch.tensor
                The generated metamer imag

## See also

[`Perception`](../../../perception.md)
