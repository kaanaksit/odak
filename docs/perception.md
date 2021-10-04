# Visual perception

The `perception` module of `odak` focuses on visual perception, and in particular on gaze-contingent perceptual loss functions. 

It contains an implementation of a metameric loss function. When used in optimisation tasks, this loss function enforces the optimised image to be a [ventral metamer](https://www.nature.com/articles/nn.2889) to the ground truth image.

This loss function is based on previous work on [fast metamer generation](https://vr-unity-viewer.cs.ucl.ac.uk/). It uses the same statistical model and many of the same acceleration techniques (e.g. MIP map sampling) to enable the metameric loss to run efficiently.

## Engineering notes

### Using the Metameric Loss Function

Our metameric loss function works in a very similar way to built in loss functions in `pytorch`, such as `torch.nn.MSELoss()`. However, it has a number of parameters which can be adjusted on creation (see the [documentation](odak/learn/perception/metameric_loss.md)). Additionally, when calculating the loss a gaze location must be specified. For example:

`loss_func = odak.learn.perception.MetamericLoss()`
`loss = loss_func(my_image, gt_image, gaze=[0.7, 0.3])`

The loss function caches some information, and performs most efficiently when repeatedly calculating losses for the same image size, with the same gaze location and foveation settings. 

We recommend adjusting the parameters of the loss function to match your application. Most importantly, please set the `real_image_width` and `real_viewing_distance` parameters to correspond to how your image will be displayed to the user. The `alpha` parameter controls the intensity of the foveation effect. You should only need to set `alpha` once - you can then adjust the width and viewing distance to achieve the same apparent foveation effect on a range of displays & viewing conditions. Note that we assume the pixels in the displayed image are square, and derive the height from the image dimensions.

We also provide two baseline loss functions `BlurLoss` and `MetamerMSELoss` which function in much the same way.

At the present time the loss functions are implemented only for images displayed to a user on a flat 2D display (e.g. an LCD computer monitor). Support for equirectangular 3D images is planned for the future.

## odak.learn.perception

| Function      | Description   |
| ------------- |:-------------:|
| [odak.learn.perception.MetamericLoss()](odak/learn/perception/metameric_loss.md) | Metameric loss function |
| [odak.learn.perception.MetamerMSELoss()](odak/learn/perception/metamer_mse_loss.md) | Metamer MSE loss function |
| [odak.learn.perception.BlurLoss()](odak/learn/perception/blur_loss.md) | Blur function |
| [odak.learn.perception.RadiallyVaryingBlur()](odak/learn/perception/radially_varying_blur.md) | Radially varying blur |
| [odak.learn.perception.SpatialSteerablePyramid()](odak/learn/perception/spatial_steerable_pyramid.md) | Spatial implementation of the real-valued steerable pyramid |
| [odak.learn.perception.make_3d_location_map()](odak/learn/perception/make_3d_location_map.md) | Foveation method: make a map of 3D locations for each pixel |
| [odak.learn.perception.make_eccentricity_distance_maps()](odak/learn/perception/make_eccentricity_distance_maps.md) | Foveation method: make maps of eccentricity and distance to the image plane for each pixel |
| [odak.learn.perception.make_pooling_size_map_pixels()](odak/learn/perception/make_pooling_size_map_pixels.md) | Foveation method: make a map of pooling sizes (in pixels) |
| [odak.learn.perception.make_pooling_size_map_lod()](odak/learn/perception/make_pooling_size_map_lod.md) | Foveation method: make a map of pooling sizes (in LoD levels) |
| [odak.learn.perception.make_radial_map()](odak/learn/perception/make_radial_map.md) | Foveation method: make a map of distances from a gaze point in pixels |
| [odak.learn.perception.ycrcb_2_rgb()](odak/learn/perception/ycrcb_2_rgb.md) | Colorspace conversion from YCrCb to RGB |
| [odak.learn.perception.rgb_2_ycrcb()](odak/learn/perception/rgb_2_ycrcb.md) | Colorspace conversion from RGB to YCrCb |


