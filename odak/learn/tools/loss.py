import torch
from .file import resize


def multi_scale_total_variation_loss(frame, levels=3):
    """
    Calculates multi-scale total variation loss for an input frame.

    This function computes the total variation loss at multiple scales by creating
    an image pyramid where each level has half the resolution of the previous level.

    Parameters
    ----------
    frame : torch.Tensor
        Input frame with shape [1 x 3 x m x n], [3 x m x n], or [m x n].
    levels : int, optional
        Number of scales in the image pyramid (default: 3).

    Returns
    -------
    torch.Tensor
        Total variation loss value.
    """
    if len(frame.shape) == 2:
        frame = frame.unsqueeze(0)
    if len(frame.shape) == 3:
        frame = frame.unsqueeze(0)
    scale = torch.nn.Upsample(scale_factor=0.5, mode="nearest")
    level = frame
    loss = 0
    for i in range(levels):
        if i != 0:
            level = scale(level)
        loss += total_variation_loss(level)
    return loss


def total_variation_loss(frame):
    """
    Calculates total variation loss for an input frame.

    This function computes the total variation loss by calculating spatial gradients
    in both x and y directions and averaging their squared values.

    Parameters
    ----------
    frame : torch.Tensor
        Input frame with shape [1 x 3 x m x n], [3 x m x n], or [m x n].

    Returns
    -------
    torch.Tensor
        Total variation loss value.
    """
    if len(frame.shape) == 2:
        frame = frame.unsqueeze(0)
    if len(frame.shape) == 3:
        frame = frame.unsqueeze(0)
    diff_x, diff_y = spatial_gradient(frame)
    pixel_count = frame.shape[0] * frame.shape[1] * frame.shape[2] * frame.shape[3]
    loss = ((diff_x**2).sum() + (diff_y**2).sum()) / pixel_count
    return loss


def spatial_gradient(frame):
    """
    Calculates the spatial gradient of a given frame.

    This function computes the gradient of the input frame in both x and y directions
    by differencing adjacent pixels.

    Parameters
    ----------
    frame : torch.Tensor
        Input frame with shape [1 x 3 x m x n], [3 x m x n], or [m x n].

    Returns
    -------
    tuple
        Tuple of (diff_x, diff_y) representing spatial gradients along x and y axes.
    """
    if len(frame.shape) == 2:
        frame = frame.unsqueeze(0)
    if len(frame.shape) == 3:
        frame = frame.unsqueeze(0)
    diff_x = frame[:, :, :, 1:] - frame[:, :, :, :-1]
    diff_y = frame[:, :, 1:, :] - frame[:, :, :-1, :]
    return diff_x, diff_y


def radial_basis_function(value, epsilon=0.5):
    """
    Applies radial basis function with Gaussian description to input values.

    This function applies the Gaussian radial basis function: y = e^(-ε² * x²)

    Parameters
    ----------
    value : torch.Tensor
        Value(s) to pass to the radial basis function.
    epsilon : float, optional
        Epsilon parameter used in the Gaussian radial basis function (default: 0.5).

    Returns
    -------
    torch.Tensor
        Output values after applying the radial basis function.
    """
    output = torch.exp((-((epsilon * value) ** 2)))
    return output


def histogram_loss(frame, ground_truth, bins=32, limits=[0.0, 1.0]):
    """
    Calculates histogram loss between input frame and ground truth.

    This function computes the MSE loss between histograms of the input frame
    and ground truth images, divided into specified number of bins.

    Parameters
    ----------
    frame : torch.Tensor
        Input frame with shape [1 x 3 x m x n], [3 x m x n], [1 x m x n], or [m x n].
    ground_truth : torch.Tensor
        Ground truth with shape [1 x 3 x m x n], [3 x m x n], [1 x m x n], or [m x n].
    bins : int, optional
        Number of bins for histogram calculation (default: 32).
    limits : list, optional
        Histogram limits as [min, max] (default: [0.0, 1.0]).

    Returns
    -------
    torch.Tensor
        Histogram loss value.
    """
    if len(frame.shape) == 2:
        frame = frame.unsqueeze(0).unsqueeze(0)
    elif len(frame.shape) == 3:
        frame = frame.unsqueeze(0)

    if len(ground_truth.shape) == 2:
        ground_truth = ground_truth.unsqueeze(0).unsqueeze(0)
    elif len(ground_truth.shape) == 3:
        ground_truth = ground_truth.unsqueeze(0)

    histogram_frame = torch.zeros(frame.shape[1], bins).to(frame.device)
    histogram_ground_truth = torch.zeros(ground_truth.shape[1], bins).to(frame.device)

    l2 = torch.nn.MSELoss()

    for i in range(frame.shape[1]):
        histogram_frame[i] = torch.histc(
            frame[:, i].flatten(), bins=bins, min=limits[0], max=limits[1]
        )
        histogram_ground_truth[i] = torch.histc(
            ground_truth[:, i].flatten(), bins=bins, min=limits[0], max=limits[1]
        )

    loss = l2(histogram_frame, histogram_ground_truth)

    return loss


def weber_contrast(image, roi_high, roi_low):
    """
    Calculates Weber contrast ratio for given regions of an image.

    This function computes the Weber contrast ratio for high and low intensity regions
    using the formula: (mean_high - mean_low) / mean_low.

    Parameters
    ----------
    image : torch.Tensor
        Input image with shape [1 x 3 x m x n], [3 x m x n], [1 x m x n], or [m x n].
    roi_high : torch.Tensor
        Corner locations of the high intensity region [m_start, m_end, n_start, n_end].
    roi_low : torch.Tensor
        Corner locations of the low intensity region [m_start, m_end, n_start, n_end].

    Returns
    -------
    torch.Tensor
        Weber contrast for the given regions. Shape is [1] or [3] depending on input.
    """
    if len(image.shape) == 2:
        image = image.unsqueeze(0)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    region_low = image[:, :, roi_low[0] : roi_low[1], roi_low[2] : roi_low[3]]
    region_high = image[:, :, roi_high[0] : roi_high[1], roi_high[2] : roi_high[3]]
    high = torch.mean(region_high, dim=(2, 3))
    low = torch.mean(region_low, dim=(2, 3))
    result = (high - low) / low
    return result.squeeze(0)


def michelson_contrast(image, roi_high, roi_low):
    """
    Calculates Michelson contrast ratio for given regions of an image.

    This function computes the Michelson contrast ratio for high and low intensity regions
    using the formula: (mean_high - mean_low) / (mean_high + mean_low).

    Parameters
    ----------
    image : torch.Tensor
        Input image with shape [1 x 3 x m x n], [3 x m x n], or [m x n].
    roi_high : torch.Tensor
        Corner locations of the high intensity region [m_start, m_end, n_start, n_end].
    roi_low : torch.Tensor
        Corner locations of the low intensity region [m_start, m_end, n_start, n_end].

    Returns
    -------
    torch.Tensor
        Michelson contrast for the given regions. Shape is [1] or [3] depending on input.
    """
    if len(image.shape) == 2:
        image = image.unsqueeze(0)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    region_low = image[:, :, roi_low[0] : roi_low[1], roi_low[2] : roi_low[3]]
    region_high = image[:, :, roi_high[0] : roi_high[1], roi_high[2] : roi_high[3]]
    high = torch.mean(region_high, dim=(2, 3))
    low = torch.mean(region_low, dim=(2, 3))
    result = (high - low) / (high + low)
    return result.squeeze(0)


def wrapped_mean_squared_error(image, ground_truth, reduction="mean"):
    """
    Calculates wrapped mean squared error between predicted and target angles.

    This function computes the mean squared error for angular data, accounting for
    the wrap-around property of angles (e.g., 359° and 1° are close).

    Parameters
    ----------
    image : torch.Tensor
        Predicted image with shape [1 x 3 x m x n], [3 x m x n], [1 x m x n], or [m x n].
    ground_truth : torch.Tensor
        Ground truth image with shape [1 x 3 x m x n], [3 x m x n], [1 x m x n], or [m x n].
    reduction : str, optional
        Specifies the reduction to apply to the output: 'mean' (default) or 'sum'.

    Returns
    -------
    torch.Tensor
        The calculated wrapped mean squared error.

    Raises
    ------
    ValueError
        If an invalid reduction type is specified.
    """
    sin_diff = torch.sin(image) - torch.sin(ground_truth)
    cos_diff = torch.cos(image) - torch.cos(ground_truth)
    loss = sin_diff**2 + cos_diff**2

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Invalid reduction type. Choose 'mean' or 'sum'.")
