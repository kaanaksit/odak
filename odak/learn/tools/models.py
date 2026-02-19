import torch


def freeze(model):
    """
    A utility function to freeze the parameters of a provided model.
    
    This function sets `requires_grad` to `False` for all parameters in the model,
    effectively freezing them during training.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters are to be frozen. This should be a PyTorch model instance.

    Returns
    -------
    None
        The function modifies the model in-place.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze(model):
    """
    A utility function to unfreeze the parameters of a provided model.
    
    This function sets `requires_grad` to `True` for all parameters in the model,
    effectively allowing them to be updated during training.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters are to be unfrozen. This should be a PyTorch model instance.

    Returns
    -------
    None
        The function modifies the model in-place.
    """
    for parameter in model.parameters():
        parameter.requires_grad = True
