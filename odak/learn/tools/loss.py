import torch


def psnr(image, ground_truth, peak_value=1.0):
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
    loss = ((diff_x**2).sum() + (diff_y**2).sum()) / pixel_count
    return loss
