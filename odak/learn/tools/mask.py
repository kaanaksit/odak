import torch
from typing import Union


def circular_binary_mask(
    px: int, py: int, r: Union[int, float], offset_x: float = 0.0, offset_y: float = 0.0
) -> torch.Tensor:
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
    offset_x : float
        Offset of the circle center in x direction (in pixels).
    offset_y : float
        Offset of the circle center in y direction (in pixels).

    Returns
    -------
    torch.Tensor
        Binary mask of shape [1, 1, px, py].
    """
    x = torch.linspace(-px / 2.0, px / 2.0, px)
    y = torch.linspace(-py / 2.0, py / 2.0, py)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    X = X - offset_x
    Y = Y - offset_y
    Z = (X**2 + Y**2) ** 0.5
    mask = torch.zeros_like(Z)
    mask[Z < r] = 1
    return mask.unsqueeze(0).unsqueeze(0)
