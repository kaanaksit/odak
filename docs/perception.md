# Visual perception

The `perception` module of `odak` focuses on visual perception, and in particular on gaze-contingent perceptual loss functions. 

## Metamers
It contains an implementation of a metameric loss function. When used in optimisation tasks, this loss function enforces the optimised image to be a [ventral metamer](https://www.nature.com/articles/nn.2889) to the ground truth image.

This loss function is based on previous work on [fast metamer generation](https://vr-unity-viewer.cs.ucl.ac.uk/). It uses the same statistical model and many of the same acceleration techniques (e.g. MIP map sampling) to enable the metameric loss to run efficiently.



## Engineering notes
| Note          | Description   |
| ------------- |:-------------:|
| [`Using metameric loss in Odak`](notes/using_metameric_loss.md) | This engineering note will give you an idea about how to use the metameric perceptual loss in Odak. |


