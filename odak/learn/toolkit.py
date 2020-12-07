from odak import np
import torch
import math

## The following functions are revised from https://github.com/computational-imaging/neural-holography
def ifftshift(tensor):
    """ifftshift for tensors of dimensions [height, width]
    shifts the width and heights
    using negative dim index for being robust to differen tensor shapes 
        especially when there is batch and channel dims.
    """
    size = tensor.size()
    if size[-1] == 2:
        first_dim = -3 
        second_dim = -2 
    else: # for tensor with torch.cfloat type
        first_dim = -2
        second_dim = -1
    tensor_shifted = roll_torch(tensor, -math.floor(size[first_dim] / 2.0), first_dim)
    tensor_shifted = roll_torch(tensor_shifted, -math.floor(size[second_dim] / 2.0), second_dim)
    return tensor_shifted


def fftshift(tensor):
    """fftshift for tensors of dimensions [height, width]
    shifts the width and heights
    using negative dim index for being robust to differen tensor shapes 
        especially when there is batch and channel dims.
    """
    size = tensor.size()
    if size[-1] == 2:
        first_dim = -3 
        second_dim = -2 
    else: # for tensor with torch.cfloat type
        first_dim = -2
        second_dim = -1
    size = tensor.size()
    tensor_shifted = roll_torch(tensor, math.floor(size[first_dim] / 2.0), first_dim)
    tensor_shifted = roll_torch(tensor_shifted, math.floor(size[second_dim] / 2.0), second_dim)
    return tensor_shifted

def roll_torch(tensor, shift, axis):
    """implements numpy roll() or Matlab circshift() functions for tensors"""
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)
