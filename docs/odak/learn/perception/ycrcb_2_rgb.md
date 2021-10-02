# odak.learn.perception.ycrcb_2_rgb

`ycrcb_2_rgb(image)`

Converts an image from YCrCb colourspace to RGB colourspace.

**Parameters:**

    image   : torch.tensor
                Input image. Should be a YCrCb floating-point image with values in the range [0, 1]
                Should be in NCHW format.

**Returns**

    rgb     : torch.tensor
                Image converted to RGB colourspace.