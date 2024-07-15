import logging
import torch.nn as nn


class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
 
    def forward(self, predictions, targets):
        """
        Args:
            predictions (Tensor): The predicted images.
            targets (Tensor): The ground truth images.

        Returns:
            float: The computed PSNR value if successful, otherwise 0.0.
        """
        try:
            from torchmetrics.functional.image import peak_signal_noise_ratio
            l_PSNR = peak_signal_noise_ratio(predictions, targets)
            return l_PSNR
        except Exception as e:
            logging.warning('PSNR failed to compute.')
            logging.warning(e)
            return 0.0
    
class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, predictions, targets):
        """
        Args:
            predictions (Tensor): The predicted images.
            targets (Tensor): The ground truth images.

        Returns:
            float: The computed SSIM value if successful, otherwise 0.0.
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
            return 0.0

class MSSSIM(nn.Module):
    def __init__(self):
        super(MSSSIM, self).__init__()

    def forward(self, predictions, targets):
        """
        Args:
            predictions (Tensor): The predicted images.
            targets (Tensor): The ground truth images.

        Returns:
            float: The computed MS-SSIM value if successful, otherwise 0.0.
        """
        try:
            from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
            if len(predictions.shape) == 3:
                predictions = predictions.unsqueeze(0)
                targets = targets.unsqueeze(0)
            l_MSSSIM = multiscale_structural_similarity_index_measure(predictions, targets, data_range=1.0)
            return l_MSSSIM  
        except Exception as e:
            logging.warning('MS-SSIM failed to compute.')
            logging.warning(e)
            return 0.0