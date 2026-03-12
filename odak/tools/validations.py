"""
odak.tools.validations

Provides validation functions for optical/simulation parameters.
Ensures robust numerical stability and device compliance across the library.
"""

import numpy as np
import torch
from ..log import logger


def validate_positive_parameter(value, name="value", allow_zero=False):
    """
    Validate that a parameter is positive (or non-negative if allow_zero).

    Parameters
    ----------
    value             : float or np.ndarray or torch.Tensor
                        Value to validate.
    name              : str
                        Parameter name for error messages.
    allow_zero        : bool
                        If True, zero values are allowed. Default: False.

    Returns
    -------
    validated_value   : same type as input
                        The validated value (unchanged if valid).

    Raises
    ------
    ValueError        : If value is negative or zero (when not allowed).
    TypeError         : If value is not a number.
    """
    if isinstance(value, (int, float)):
        if not allow_zero and value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        elif allow_zero and value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    elif isinstance(value, np.ndarray):
        if not allow_zero and np.any(value <= 0):
            raise ValueError(f"{name} contains negative or zero values")
        elif allow_zero and np.any(value < 0):
            raise ValueError(f"{name} contains negative values")

    elif isinstance(value, torch.Tensor):
        if not allow_zero and torch.any(value <= 0):
            raise ValueError(f"{name} contains negative or zero values")
        elif allow_zero and torch.any(value < 0):
            raise ValueError(f"{name} contains negative values")

    else:
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")

    logger.debug(
        f"Validated {name}: {value if isinstance(value, (int, float)) else f'{value.shape}'}"
    )
    return value


def validate_wavelength(wavelength, allow_zero=False):
    """
    Validate wavelength parameter for optical calculations.

    Parameters
    ----------
    wavelength        : float
                        Wavelength in mm. Must be positive.
    allow_zero        : bool
                        If True, zero allowed (not recommended). Default: False.

    Returns
    -------
    wavelength        : float
                        Validated wavelength value.

    Raises
    ------
    ValueError        : If wavelength is zero or negative.
    """
    return validate_positive_parameter(wavelength, "wavelength", allow_zero)


def validate_distance(distance, name="distance"):
    """
    Validate propagation distance parameter.

    Parameters
    ----------
    distance          : float
                        Propagation/observation distance in mm. Must be positive.
    name              : str
                        Custom name for error messages.

    Returns
    -------
    distance          : float
                        Validated distance value.

    Raises
    ------
    ValueError        : If distance is zero or negative.
    """
    return validate_positive_parameter(distance, name)


def validate_pixel_pitch(pixel_pitch, allow_zero=False):
    """
    Validate pixel pitch or spatial sampling parameter (dx).

    Parameters
    ----------
    pixel_pitch       : float
                        Pixel size/pitch in mm. Must be positive.
    allow_zero        : bool
                        If True, zero allowed (not recommended). Default: False.

    Returns
    -------
    pixel_pitch       : float
                        Validated pixel pitch value.

    Raises
    ------
    ValueError        : If pixel_pitch is zero or negative.
    """
    return validate_positive_parameter(pixel_pitch, "pixel_pitch", allow_zero)


def validate_dimensions(shape=None, tensor=None, required_dims=None):
    """
    Validate tensor/array dimensions for optical operations.

    Parameters
    ----------
    shape             : tuple or None
                        Shape to validate.
    tensor            : torch.Tensor or np.ndarray or None
                        Tensor to extract shape from.
    required_dims     : int or tuple or None
                        Required dimensions.

    Returns
    -------
    message           : str
                        Validation result message.

    Raises
    ------
    ValueError        : If dimension validation fails.
    """
    if tensor is not None:
        if isinstance(tensor, torch.Tensor):
            dims = tensor.dim()
            actual_shape = tensor.shape
        elif isinstance(tensor, np.ndarray):
            dims = tensor.ndim
            actual_shape = tensor.shape
        else:
            raise TypeError(f"Tensor must be torch.Tensor or np.ndarray")
    elif shape is not None:
        dims = len(shape)
        actual_shape = shape
    else:
        raise ValueError("Either 'tensor' or 'shape' must be provided")

    if required_dims is not None:
        if isinstance(required_dims, int):
            if dims != required_dims:
                raise ValueError(
                    f"Expected {required_dims}D tensor, got {dims}D with shape {actual_shape}"
                )
        elif isinstance(required_dims, tuple) and len(required_dims) == 2:
            min_d, max_d = required_dims
            if dims < min_d or dims > max_d:
                raise ValueError(
                    f"Expected {min_d}-{max_d}D tensor, got {dims}D with shape {actual_shape}"
                )

    message = f"Validated {dims}D {'torch.Tensor' if isinstance(tensor, torch.Tensor) else 'array'} shape: {actual_shape}"
    logger.debug(message)
    return message


def validate_device(device=None):
    """
    Validate and determine device placement for tensor operations.

    Ensures consistent CPU vs GPU usage. Per AGENTS.md requirements:
    'All tensor operations should handle device placement explicitly (CPU vs CUDA)'

    Parameters
    ----------
    device            : torch.device or str or None
                        Device specification (None = auto-detect).

    Returns
    -------
    device            : torch.device
                        Validated device object.
    message           : str
                        Device placement information.

    Examples
    --------
    >>> dev, msg = validate_device('cpu')
    > (device(type='cpu'), 'Using CPU device')
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            message = f"Auto-detected GPU: {torch.cuda.get_device_name(device)}"
            logger.info(message)
        else:
            device = torch.device("cpu")
            message = "Using CPU (CUDA not available)"
    elif isinstance(device, str):
        if device.lower() == "cpu":
            device = torch.device("cpu")
            message = "Forced execution on CPU"
        elif device.lower().startswith("cuda"):
            if torch.cuda.is_available():
                if ":" in device:
                    idx = int(device.split(":")[1])
                    if idx < torch.cuda.device_count():
                        device = torch.device("cuda", idx)
                        message = (
                            f"Using GPU #{idx}: {torch.cuda.get_device_name(device)}"
                        )
                    else:
                        raise ValueError(f"GPU index {idx} out of range")
                else:
                    device = torch.device("cuda")
                    message = f"Using GPU: {torch.cuda.get_device_name(device)}"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = torch.device("cpu")
                message = "Fallback to CPU (CUDA unavailable)"
        else:
            raise ValueError(
                f"Unknown device: {device}. Use 'cpu', 'cuda', or torch.device"
            )
    elif isinstance(device, torch.device):
        message = f"Using provided device: {device}"
    else:
        raise TypeError(f"Device must be str, torch.device, or None")

    logger.debug(message)
    return device, message
