import torch


def freeze(model):
    """
    A utility function to freeze the parameters of a provided model defined as a Pythonic class.
    For instance, `odak.learn.models.unet` is such a model.

    Parameters
    ----------
    model          : torch.nn.modules
                     Model to be frozen, in other terms `requires_grad` to be set to `False`.                    
    """
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze(model):
    """
    A utility function to unfreeze the parameters of a provided model defined as a Pythonic class.
    For instance, `odak.learn.models.unet` is such a model.


    Parameters
    ----------
    model          : torch.nn.modules
                     Model to unfreeze, in other terms `requires_grad` to be set to `True`.
    """
    for parameter in model.parameters():
        parameter.requires_grad = True
