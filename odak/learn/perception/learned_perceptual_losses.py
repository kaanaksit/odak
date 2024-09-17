import logging
import torch
import torch.nn as nn


class CVVDP(nn.Module):
    def __init__(self, device = torch.device('cpu')):
        """
        Initializes the CVVDP model with a specified device.

        Parameters
        ----------
        device   : torch.device
                    The device (CPU/GPU) on which the computations will be performed. Defaults to CPU.
        """
        super(CVVDP, self).__init__()
        try:
            import pycvvdp
            self.cvvdp = pycvvdp.cvvdp(display_name = 'standard_4k', device = device)
        except Exception as e:
            logging.warning('ColorVideoVDP is missing, consider installing by running "pip install -U git+https://github.com/gfxdisp/ColorVideoVDP"')
            logging.warning(e)


    def forward(self, predictions, targets, dim_order = 'CHW'):
        """
        Parameters
        ----------
        predictions   : torch.tensor
                        The predicted images.
        targets    h  : torch.tensor
                        The ground truth images.
        dim_order     : str
                        The dimension order of the input images. Defaults to 'CHW' (channels, height, width).

        Returns
        -------
        result        : torch.tensor
                        The computed loss if successful, otherwise 0.0.
        """
        try:
            l_ColorVideoVDP = self.cvvdp.loss(predictions, targets, dim_order = dim_order)
            return l_ColorVideoVDP
        except Exception as e:
            logging.warning('ColorVideoVDP failed to compute.')
            logging.warning(e)
            return torch.tensor(0.0)
        
class FVVDP(nn.Module):
    def __init__(self, device = torch.device('cpu')):
        """
        Initializes the FVVDP model with a specified device.

        Parameters
        ----------
        device   : torch.device
                    The device (CPU/GPU) on which the computations will be performed. Defaults to CPU.
        """
        super(FVVDP, self).__init__()
        try:
            import pyfvvdp
            self.fvvdp = pyfvvdp.fvvdp(display_name = 'standard_4k', heatmap = 'none', device = device)
        except Exception as e:
            logging.warning('FovVideoVDP is missing, consider installing by running "pip install pyfvvdp"')
            logging.warning(e)


    def forward(self, predictions, targets, dim_order = 'CHW'):
        """
        Parameters
        ----------
        predictions   : torch.tensor
                        The predicted images.
        targets       : torch.tensor
                        The ground truth images.
        dim_order     : str
                        The dimension order of the input images. Defaults to 'CHW' (channels, height, width).

        Returns
        -------
        result        : torch.tensor
                          The computed loss if successful, otherwise 0.0.
        """
        try:
            l_FovVideoVDP = self.fvvdp.predict(predictions, targets, dim_order = dim_order)[0]
            return l_FovVideoVDP
        except Exception as e:
            logging.warning('FovVideoVDP failed to compute.')
            logging.warning(e)
            return torch.tensor(0.0)


class LPIPS(nn.Module):

    def __init__(self):
        """
        Initializes the LPIPS (Learned Perceptual Image Patch Similarity) model.

        """
        super(LPIPS, self).__init__()
        try:
            import torchmetrics
            self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type = 'squeeze')
        except Exception as e:
            logging.warning('torchmetrics is missing, consider installing by running "pip install torchmetrics"')
            logging.warning(e)


    def forward(self, predictions, targets):
        """
        Parameters
        ----------
        predictions   : torch.tensor
                        The predicted images.
        targets       : torch.tensor
                        The ground truth images.
       
        Returns
        -------
        result        : torch.tensor
                        The computed loss if successful, otherwise 0.0.
        """
        try:
            lpips_image = predictions
            lpips_target = targets
            if len(lpips_image.shape) == 3:
                lpips_image = lpips_image.unsqueeze(0)
                lpips_target = lpips_target.unsqueeze(0)
            if lpips_image.shape[1] == 1:
                lpips_image = lpips_image.repeat(1, 3, 1, 1)
                lpips_target = lpips_target.repeat(1, 3, 1, 1)
            lpips_image = (lpips_image * 2 - 1).clamp(-1, 1)
            lpips_target = (lpips_target * 2 - 1).clamp(-1, 1)
            l_LPIPS = self.lpips(lpips_image, lpips_target)
            return l_LPIPS
        except Exception as e:
            logging.warning('LPIPS failed to compute.')
            logging.warning(e)
            return torch.tensor(0.0)
           
