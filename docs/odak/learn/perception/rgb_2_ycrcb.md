# odak.learn.perception.rgb_2_ycrcb

`rgb_2_ycrcb(image)`

Converts an image from RGB colourspace to YCrCb colourspace.

**Parameters:**

    image   : torch.tensor
                Input image. Should be an RGB floating-point image with values in the range [0, 1]
                Should be in NCHW format.

**Returns**

    ycrcb   : torch.tensor
                Image converted to YCrCb colourspace.