"""
``odak.learn.lensless``
 
Defines a shallow neural network for rotational pose retrieval.

This module provides the SpecTrack model which is used for multi-rotation tracking 
via speckle imaging, as described in the paper "SpecTrack: Learned Multi-Rotation 
Tracking via Speckle Imaging" by Ziyang Chen, Mustafa Dogan, Josef Spjut, and Kaan Akşit.
The model is a deep neural network that takes speckle images as input and predicts
3D rotation angles.

Example usage:
    >>> import odak.learn.lensless as lensless
    >>> model = lensless.spec_track()
    >>> # Use model for training or inference

"""
from .models import spec_track
