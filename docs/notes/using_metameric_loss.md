This engineering note will give you an idea about using the metameric perceptual loss in `odak`. 
This note is compiled by `David Walton`. 
If you have further questions regarding this note, please email `David` at `david.walton.13@ucl.ac.uk`.

Our metameric loss function works in a very similar way to built in loss functions in `pytorch`, such as `torch.nn.MSELoss()`. 
However, it has a number of parameters which can be adjusted on creation (see the [documentation](../../odak/learn/perception/metameric_loss)). 
Additionally, when calculating the loss a gaze location must be specified. For example:

```
loss_func = odak.learn.perception.MetamericLoss()
loss = loss_func(my_image, gt_image, gaze=[0.7, 0.3])
```

The loss function caches some information, and performs most efficiently when repeatedly calculating losses for the same image size, with the same gaze location and foveation settings.

We recommend adjusting the parameters of the loss function to match your application. 
Most importantly, please set the `real_image_width` and `real_viewing_distance` parameters to correspond to how your image will be displayed to the user. 
The `alpha` parameter controls the intensity of the foveation effect. 
You should only need to set `alpha` once - you can then adjust the width and viewing distance to achieve the same apparent foveation effect on a range of displays & viewing conditions.
Note that we assume the pixels in the displayed image are square, and derive the height from the image dimensions.

We also provide two baseline loss functions `BlurLoss` and `MetamerMSELoss` which function in much the same way.

At the present time the loss functions are implemented only for images displayed to a user on a flat 2D display (e.g. an LCD computer monitor). 
Support for equirectangular 3D images is planned for the future.

## See also

[`Visual perception`](../../perception)
