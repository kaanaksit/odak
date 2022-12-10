import torch
from .file import resize


def psnr(image, ground_truth, peak_value = 1.0):
    """
    A function to calculate peak-signal-to-noise ratio of an image with respect to a ground truth image.

    Parameters
    ----------
    image         : torch.tensor
                    Image to be tested.
    ground_truth  : torch.tensor
                    Ground truth image.
    peak_value    : float
                    Peak value that given tensors could have.

    Returns
    -------
    result        : torch.tensor
                    Peak-signal-to-noise ratio.
    """
    mse = torch.mean((ground_truth - image)**2)
    result = 20 * torch.log10(peak_value / torch.sqrt(mse))
    return result


def multi_scale_total_variation_loss(frame, levels = 3):
    """
    Function for evaluating a frame against a target using multi scale total variation approach. Here, multi scale refers to image pyramid of an input frame, where at each level image resolution is half of the previous level.
        
    Parameters
    ----------
    frame         : torch.tensor
                    Input frame [1 x 3 x m x n] or [3 x m x n] or [m x n].
    levels        : int
                    Number of levels to go in the image pyriamid.

    Returns
    -------
    loss          : float
                    Loss from evaluation.
    """
    if len(frame.shape) == 2:
        frame = frame.unsqueeze(0)
    if len(frame.shape) == 3:
        frame = frame.unsqueeze(0)
    scale = torch.nn.Upsample(scale_factor = 0.5, mode = 'nearest')
    level = frame
    loss = 0
    for i in range(levels):
        if i != 0:
           level = scale(level)
        loss += total_variation_loss(level) 
    return loss


def total_variation_loss(frame):
    """
    Function for evaluating a frame against a target using total variation approach.
        
    Parameters
    ----------
    frame         : torch.tensor
                    Input frame [1 x 3 x m x n] or [3 x m x n] or [m x n].

    Returns
    -------
    loss          : float
                    Loss from evaluation.
    """
    if len(frame.shape) == 2:
        frame = frame.unsqueeze(0)
    if len(frame.shape) == 3:
        frame = frame.unsqueeze(0)
    diff_x = frame[:, :, :, 1:] - frame[:, :, :, :-1]
    diff_y = frame[:, :, 1:, :] - frame[:, :, :-1, :]
    pixel_count = frame.shape[0] * frame.shape[1] * frame.shape[2] * frame.shape[3]
    loss = ((diff_x ** 2).sum() + (diff_y ** 2).sum()) / pixel_count
    return loss


def radial_basis_function(value, epsilon = 0.5):
    """
    Function to pass a value into radial basis function with Gaussian description.

    Parameters
    ----------
    value            : torch.tensor
                       Value(s) to pass to the radial basis function. 
    epsilon          : float
                       Epsilon used in the Gaussian radial basis function (e.g., y=e^(-(epsilon x value)^2).

    Returns
    -------
    output           : torch.tensor
                       Output values.
    """
    output = torch.exp((-(epsilon * value)**2))
    return output


def histogram_loss(frame, ground_truth, bins = 32, limits = [0., 1.]):
    """
    Function for evaluating a frame against a target using histogram.

    Parameters
    ----------
    frame            : torch.tensor
                       Input frame [1 x 3 x m x n]  or [3 x m x n] or [1 x m x n] or [m x n].
    ground_truth     : torch.tensor
                       Ground truth [1 x 3 x m x n] or  [3 x m x n] or [1 x m x n] or  [m x n].
    bins             : int
                       Number of bins.
    limits           : list
                       Limits.
    
    Returns
    -------
    loss             : float
                       Loss from evaluation.
    """
    if len(frame.shape) == 2:
        frame = frame.unsqueeze(0)
    if len(frame.shape) == 3:
        frame = frame.unsqueeze(0)
    histogram_frame = torch.zeros(frame.shape[1], bins).to(frame.device)
    histogram_ground_truth = torch.zeros(frame.shape[1], bins).to(frame.device)
    l2 = torch.nn.MSELoss()
    for i in range(frame.shape[1]):
        histogram_frame[i] = torch.histc(frame[:, i].flatten(), bins = bins, min = limits[0], max = limits[1])
        histogram_ground_truth[i] = torch.histc(frame[:, i].flatten(), bins = bins, min = limits[0], max = limits[1])
    loss = l2(histogram_frame, histogram_ground_truth)
    return loss
