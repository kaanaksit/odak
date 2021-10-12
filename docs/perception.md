# Visual perception

The `perception` module of `odak` focuses on visual perception, and in particular on gaze-contingent perceptual loss functions. 

It contains an implementation of a metameric loss function. When used in optimisation tasks, this loss function enforces the optimised image to be a [ventral metamer](https://www.nature.com/articles/nn.2889) to the ground truth image.

This loss function is based on previous work on [fast metamer generation](https://vr-unity-viewer.cs.ucl.ac.uk/). It uses the same statistical model and many of the same acceleration techniques (e.g. MIP map sampling) to enable the metameric loss to run efficiently.

## Engineering notes
| Note          | Description   |
| ------------- |:-------------:|
| [`Using metameric loss in Odak`](notes/using_metameric_loss.md) | This engineering note will give you an idea about how to use the metameric perceptual loss in Odak. |

## odak.learn.perception

| Function      | Description   |
| ------------- |:-------------:|
| [odak.learn.perception.MetamericLoss()](odak/learn/perception/metameric_loss.md) | Metameric loss function |
| [odak.learn.perception.MetamericLossUniform()](odak/learn/perception/metameric_loss_uniform.md) | Metameric loss function (uniform, not foveated variant) |
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


