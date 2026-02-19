import torch
import os
import odak.tools
from ...tools import expanduser


def resize(image, multiplier=0.5, mode="nearest"):
    """
    Definition to resize an image.

    Parameters
    ----------
    image       : torch.tensor
                  Image with MxNx3 resolution.
    multiplier  : float
                  Multiplier used in resizing operation (e.g., 0.5 is half size in one axis).
    mode        : str
                  Mode to be used in scaling, nearest, bilinear, etc.

    Returns
    -------
    new_image   : torch.tensor
                  Resized image.
    """
    # Handle the case where image needs to be in the right format for torch.nn.Upsample
    if len(image.shape) == 3:
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        image = image.unsqueeze(0)
    elif len(image.shape) == 4:
        # Image is already in batch format
        pass
    else:
        raise ValueError("Image must have 3 or 4 dimensions")
    
    # Use torch.nn.functional.interpolate for resizing
    if mode not in ["nearest", "bilinear", "bicubic", "area"]:
        raise ValueError("Mode must be one of: nearest, bilinear, bicubic, area")
    
    # Resize the image
    new_image = torch.nn.functional.interpolate(
        image, 
        scale_factor=multiplier, 
        mode=mode,
        align_corners=None if mode in ["nearest", "area"] else False
    )
    
    # Remove batch dimension if it was added
    if new_image.shape[0] == 1:
        new_image = new_image.squeeze(0)
    
    return new_image


def load_image(fn, normalizeby=0.0, torch_style=False):
    """
    Definition to load an image from a given location as a torch tensor.

    Parameters
    ----------
    fn           : str
                   Filename.
    normalizeby  : float or optional
                   Value to to normalize images with. Default value of zero will lead to no normalization.
    torch_style  : bool or optional
                   If set True, it will load an image mxnx3 as 3xmxn.

    Returns
    -------
    image        : torch.tensor
                   Image loaded as a torch tensor.
    """
    image = odak.tools.load_image(fn, normalizeby=normalizeby, torch_style=torch_style)
    image = torch.from_numpy(image).float()
    return image


def save_image(fn, img, cmin=0, cmax=255, color_depth=8):
    """
    Definition to save a torch tensor as an image.

    Parameters
    ----------
    fn           : str
                   Filename.
    img          : torch.tensor
                   A torch tensor with NxMx3 or NxMx1 shapes.
    cmin         : int
                   Minimum value that will be interpreted as 0 level in the final image.
    cmax         : int
                   Maximum value that will be interpreted as 255 level in the final image.
    color_depth  : int
                   Color depth of an image. Default is eight.

    Returns
    -------
    bool         : bool
                   True if successful.
    """
    if len(img.shape) == 4:
        img = img.squeeze(0)
    if len(img.shape) > 2 and torch.argmin(torch.tensor(img.shape)) == 0:
        # Transpose from (C, H, W) to (H, W, C) 
        new_img = torch.zeros(img.shape[1], img.shape[2], img.shape[0]).to(img.device)
        for i in range(img.shape[0]):
            new_img[:, :, i] = img[i].detach().clone()
        img = new_img.detach().clone()
    img = img.cpu().detach().numpy()
    return odak.tools.save_image(fn, img, cmin=cmin, cmax=cmax, color_depth=color_depth)


def save_torch_tensor(fn, tensor):
    """
    Definition to save a torch tensor.

    Parameters
    ----------
    fn       : str
               Filename.
    tensor   : torch.tensor
               Torch tensor to be saved.
    """
    torch.save(tensor, expanduser(fn))


def torch_load(fn, weights_only=True, map_location=None):
    """
    Definition to load a torch files (*.pt).

    Parameters
    ----------
    fn           : str
                   Filename.
    weights_only : bool
                   See torch.load() for details.
    map_location : str
                   The device location to place data (e.g., `cuda`, `cpu`, etc.).
                   The default is None.

    Returns
    -------
    data         : any
                   See torch.load() for more.
    """
    data = torch.load(
        expanduser(fn),
        weights_only=weights_only,
        map_location=map_location,
    )
    return data
