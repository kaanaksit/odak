# General toolkit.
Odak provides a set of functions that can be used for general purpose work, such as saving an image file or loading a three-dimensional point cloud of an object.
These functions are helpful for general use and provide consistency across routine works in loading and saving routines.
When working with odak, we strongly suggest sticking to the general toolkit to provide a coherent solution to your task.

## odak.tools
This submodule is based on `numpy`. If you are using functions outside of `odak.learn` submodule, we recommend you to use this specific set of tools.

| Function      | Description   |
| ------------- |:-------------:|
| [odak.tools.save_image](odak/tools/save_image.md) | Save as an image. |
