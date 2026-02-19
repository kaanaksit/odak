import torch
from typing import Union

def circular_binary_mask(px: int, py: int, r: Union[int, float]) -> torch.Tensor:
    """
    Generate a 2D circular binary mask.
    
    Parameters
    ----------
    px : int
        Pixel count in x dimension.
    py : int
        Pixel count in y dimension.
    r : Union[int, float]
        Radius of the circle.
        
    Returns
    -------
    torch.Tensor
        Binary mask of shape [1, 1, px, py].
    """
    x = torch.linspace(-px / 2.0, px / 2.0, px)
    y = torch.linspace(-py / 2.0, py / 2.0, py)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    Z = (X**2 + Y**2) ** 0.5
    mask = torch.zeros_like(Z)
    mask[Z < r] = 1
    return mask.unsqueeze(0).unsqueeze(0)
