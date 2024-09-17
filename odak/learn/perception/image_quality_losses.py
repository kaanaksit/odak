import logging
import torch
import torch.nn as nn


class PSNR(nn.Module):
    '''
    A class to calculate peak-signal-to-noise ratio of an image with respect to a ground truth image.
    '''

    def __init__(self):
        super(PSNR, self).__init__()
 
    def forward(self, predictions, targets, peak_value = 1.0):
        """
        A function to calculate peak-signal-to-noise ratio of an image with respect to a ground truth image.

        Parameters
        ----------
        predictions   : torch.tensor
                        Image to be tested.
        targets       : torch.tensor
                        Ground truth image.
        peak_value    : float
                        Peak value that given tensors could have.

        Returns
        -------
        result        : torch.tensor
                        Peak-signal-to-noise ratio.
        """
        mse = torch.mean((targets - predictions) ** 2)
        result = 20 * torch.log10(peak_value / torch.sqrt(mse))
        return result


class SSIM(nn.Module):
    '''
    A class to calculate structural similarity index of an image with respect to a ground truth image.
    '''

    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, predictions, targets):
        """
        Parameters
        ----------
        predictions : torch.tensor
                      The predicted images.
        targets     : torch.tensor
                      The ground truth images.

        Returns
        -------
        result      : torch.tensor 
                      The computed SSIM value if successful, otherwise 0.0.
        """
        try:
            from torchmetrics.functional.image import structural_similarity_index_measure
            if len(predictions.shape) == 3:
                predictions = predictions.unsqueeze(0)
                targets = targets.unsqueeze(0)
            l_SSIM = structural_similarity_index_measure(predictions, targets)
            return l_SSIM
        except Exception as e:
            logging.warning('SSIM failed to compute.')
            logging.warning(e)
            return torch.tensor(0.0)

class MSSSIM(nn.Module):
    '''
    A class to calculate multi-scale structural similarity index of an image with respect to a ground truth image.
    '''

    def __init__(self):
        super(MSSSIM, self).__init__()

    def forward(self, predictions, targets):
        """
        Parameters
        ----------
        predictions : torch.tensor
                      The predicted images.
        targets     : torch.tensor
                      The ground truth images.

        Returns
        -------
        result      : torch.tensor 
                      The computed MS-SSIM value if successful, otherwise 0.0.
        """
        try:
            from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
            if len(predictions.shape) == 3:
                predictions = predictions.unsqueeze(0)
                targets = targets.unsqueeze(0)
            l_MSSSIM = multiscale_structural_similarity_index_measure(predictions, targets, data_range = 1.0)
            return l_MSSSIM  
        except Exception as e:
            logging.warning('MS-SSIM failed to compute.')
            logging.warning(e)
            return torch.tensor(0.0)
