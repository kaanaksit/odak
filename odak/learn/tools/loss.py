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
