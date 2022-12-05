import torch


def circular_binary_mask(px, py, r):
    """
    Definition to generate a 2D circular binary mask.

    Parameter
    ---------
    px           : int
                   Pixel count in x.
    py           : int
                   Pixel count in y.
    r            : int
                   Radius of the circle.
    
    Returns
    -------
    mask         : torch.tensor
                   Mask [1 x 1 x m x n].
    """
    x = torch.linspace(-px / 2., px / 2., px)
    y = torch.linspace(-py / 2., py / 2., py)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = (X ** 2 + Y ** 2) ** 0.5
    mask = torch.zeros_like(Z)
    mask[Z < r] = 1
    return mask
