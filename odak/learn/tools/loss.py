import torch
from .file import resize


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
        histogram_frame[i] = torch.histc(frame[:, i].flatten(), bins=bins, min=limits[0], max=limits[1])
        histogram_ground_truth[i] = torch.histc(ground_truth[:, i].flatten(), bins=bins, min=limits[0], max=limits[1])
    
    loss = l2(histogram_frame, histogram_ground_truth)
    
    return loss

    
def weber_contrast(image, roi_high, roi_low):
    """
    A function to calculate weber contrast ratio of given region of interests of the image.

    Parameters
    ----------
    image         : torch.tensor
                    Image to be tested [1 x 3 x m x n] or [3 x m x n] or [1 x m x n] or [m x n].
    roi_high      : torch.tensor
                    Corner locations of the roi for high intensity area [m_start, m_end, n_start, n_end].
    roi_low       : torch.tensor
                    Corner locations of the roi for low intensity area [m_start, m_end, n_start, n_end].

    Returns
    -------
    result        : torch.tensor
                    Weber contrast for given regions. [1] or [3] depending on input image.
    """
    if len(image.shape) == 2:
        image = image.unsqueeze(0)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    region_low = image[:, :, roi_low[0]:roi_low[1], roi_low[2]:roi_low[3]]
    region_high = image[:, :, roi_high[0]:roi_high[1], roi_high[2]:roi_high[3]]
    high = torch.mean(region_high, dim = (2, 3))
    low = torch.mean(region_low, dim = (2, 3))
    result = (high - low) / low
    return result.squeeze(0)
    
 
def michelson_contrast(image, roi_high, roi_low):
    """
    A function to calculate michelson contrast ratio of given region of interests of the image.

    Parameters
    ----------
    image         : torch.tensor
                    Image to be tested [1 x 3 x m x n] or [3 x m x n] or [m x n].
    roi_high      : torch.tensor
                    Corner locations of the roi for high intensity area [m_start, m_end, n_start, n_end].
    roi_low       : torch.tensor
                    Corner locations of the roi for low intensity area [m_start, m_end, n_start, n_end].

    Returns
    -------
    result        : torch.tensor
                    Michelson contrast for the given regions. [1] or [3] depending on input image.
    """
    if len(image.shape) == 2:
        image = image.unsqueeze(0)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    region_low = image[:, :, roi_low[0]:roi_low[1], roi_low[2]:roi_low[3]]
    region_high = image[:, :, roi_high[0]:roi_high[1], roi_high[2]:roi_high[3]]
    high = torch.mean(region_high, dim = (2, 3))
    low = torch.mean(region_low, dim = (2, 3))
    result = (high - low) / (high + low)
    return result.squeeze(0)


def wrapped_mean_squared_error(image, ground_truth, reduction = 'mean'):
    """
    A function to calculate the wrapped mean squared error between predicted and target angles.
    
    Parameters
    ----------
    image         : torch.tensor
                    Image to be tested [1 x 3 x m x n]  or [3 x m x n] or [1 x m x n] or [m x n].
    ground_truth  : torch.tensor
                    Ground truth to be tested [1 x 3 x m x n]  or [3 x m x n] or [1 x m x n] or [m x n].
    reduction     : str
                    Specifies the reduction to apply to the output: 'mean' (default) or 'sum'.

    Returns
    -------
    wmse        : torch.tensor
                  The calculated wrapped mean squared error. 
    """
    sin_diff = torch.sin(image) - torch.sin(ground_truth)
    cos_diff = torch.cos(image) - torch.cos(ground_truth)
    loss = (sin_diff**2 + cos_diff**2)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError("Invalid reduction type. Choose 'mean' or 'sum'.")