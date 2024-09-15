import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from torch.autograd import Variable
from ..tools import generate_2d_gaussian
from ..perception.learned_perceptual_losses import CVVDP, FVVDP, LPIPS
from ..perception.image_quality_losses import PSNR, SSIM, MSSSIM


class phase_gradient(nn.Module):
    
    """
    The class 'phase_gradient' provides a regularization function to measure the variation(Gradient or Laplace) of the phase of the complex amplitude. 

    This implements a convolution of the phase with a kernel.

    The kernel is a simple 3 by 3 Laplacian kernel here, but you can also try other edge detection methods.
    """
    

    def __init__(self, kernel = None, loss = nn.MSELoss(), device = torch.device("cpu")):
        """
        Parameters
        ----------
        kernel                  : torch.tensor
                                    Convolution filter kernel, 3 by 3 Laplacian kernel by default.
        loss                    : torch.nn.Module
                                    loss function, L2 Loss by default.
        """
        super(phase_gradient, self).__init__()
        self.device = device
        self.loss = loss
        if kernel == None:
            self.kernel = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32) / 8
        else:
            if len(kernel.shape) == 4:
                self.kernel = kernel
            else:
                self.kernel = kernel.reshape((1, 1, kernel.shape[0], kernel.shape[1]))
        self.kernel = Variable(self.kernel.to(self.device))
        

    def forward(self, phase):
        """
        Calculates the phase gradient Loss.

        Parameters
        ----------
        phase                  : torch.tensor
                                    Phase of the complex amplitude.

        Returns
        -------

        loss_value              : torch.tensor
                                    The computed loss.
        """

        if len(phase.shape) == 2:
            phase = phase.reshape((1, 1, phase.shape[0], phase.shape[1]))
        edge_detect = self.functional_conv2d(phase)
        loss_value = self.loss(edge_detect, torch.zeros_like(edge_detect))
        return loss_value


    def functional_conv2d(self, phase):
        """
        Calculates the gradient of the phase.

        Parameters
        ----------
        phase                  : torch.tensor
                                    Phase of the complex amplitude.

        Returns
        -------

        edge_detect              : torch.tensor
                                    The computed phase gradient.
        """
        edge_detect = F.conv2d(phase, self.kernel, padding = self.kernel.shape[-1] // 2)
        return edge_detect



class speckle_contrast(nn.Module):

    """
    The class 'speckle_contrast' provides a regularization function to measure the speckle contrast of the intensity of the complex amplitude using C=sigma/mean. Where C is the speckle contrast, mean and sigma are mean and standard deviation of the intensity.

    We refer to the following paper:

    Kim et al.(2020). Light source optimization for partially coherent holographic displays with consideration of speckle contrast, resolution, and depth of field. Scientific Reports. 10. 18832. 10.1038/s41598-020-75947-0. 
    """
    

    def __init__(self, kernel_size = 11, step_size = (1, 1), loss = nn.MSELoss(), device=torch.device("cpu")):
        """
        Parameters
        ----------
        kernel_size             : torch.tensor
                                    Convolution filter kernel size, 11 by 11 average kernel by default.
        step_size               : tuple
                                    Convolution stride in height and width direction.
        loss                    : torch.nn.Module
                                    loss function, L2 Loss by default.
        """
        super(speckle_contrast, self).__init__()
        self.device = device
        self.loss = loss
        self.step_size = step_size
        self.kernel_size = kernel_size
        self.kernel = torch.ones((1, 1, self.kernel_size, self.kernel_size)) / (self.kernel_size ** 2)
        self.kernel = Variable(self.kernel.type(torch.FloatTensor).to(self.device))


    def forward(self, intensity):
        """
        Calculates the speckle contrast Loss.

        Parameters
        ----------
        intensity               : torch.tensor
                                    intensity of the complex amplitude.

        Returns
        -------

        loss_value              : torch.tensor
                                    The computed loss.
        """

        if len(intensity.shape) == 2:
            intensity = intensity.reshape((1, 1, intensity.shape[0], intensity.shape[1]))
        Speckle_C = self.functional_conv2d(intensity)
        loss_value = self.loss(Speckle_C, torch.zeros_like(Speckle_C))
        return loss_value


    def functional_conv2d(self, intensity):
        """
        Calculates the speckle contrast of the intensity.

        Parameters
        ----------
        intensity                : torch.tensor
                                    Intensity of the complex field.

        Returns
        -------

        Speckle_C               : torch.tensor
                                    The computed speckle contrast.
        """
        mean = F.conv2d(intensity, self.kernel, stride = self.step_size)
        var = torch.sqrt(F.conv2d(torch.pow(intensity, 2), self.kernel, stride = self.step_size) - torch.pow(mean, 2))
        Speckle_C = var / mean
        return Speckle_C


class multiplane_loss():
    """
    Loss function for computing loss in multiplanar images. Unlike, previous methods, this loss function accounts for defocused parts of an image.
    """

    def __init__(self, target_image, target_depth, blur_ratio = 0.25, 
                 target_blur_size = 10, number_of_planes = 4, weights = [1., 2.1, 0.6], 
                 multiplier = 1., scheme = 'defocus', reduction = 'mean', device = torch.device('cpu')):
        """
        Parameters
        ----------
        target_image      : torch.tensor
                            Color target image [3 x m x n].
        target_depth      : torch.tensor
                            Monochrome target depth, same resolution as target_image.
        target_blur_size  : int
                            Maximum target blur size.
        blur_ratio        : float
                            Blur ratio, a value between zero and one.
        number_of_planes  : int
                            Number of planes.
        weights           : list
                            Weights of the loss function.
        multiplier        : float
                            Multiplier to multipy with targets.
        scheme            : str
                            The type of the loss, `naive` without defocus or `defocus` with defocus.
        reduction         : str
                            Reduction can either be 'mean', 'none' or 'sum'. For more see: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
        device            : torch.device
                            Device to be used (e.g., cuda, cpu, opencl).
        """
        self.device = device
        self.target_image     = target_image.float().to(self.device)
        self.target_depth     = target_depth.float().to(self.device)
        self.target_blur_size = target_blur_size
        if self.target_blur_size % 2 == 0:
            self.target_blur_size += 1
        self.number_of_planes = number_of_planes
        self.multiplier       = multiplier
        self.weights          = weights
        self.reduction        = reduction
        self.blur_ratio       = blur_ratio
        self.set_targets()
        if scheme == 'defocus':
            self.add_defocus_blur()
        self.loss_function = torch.nn.MSELoss(reduction = self.reduction)
        
    def get_targets(self):
        """
        Returns
        -------
        targets           : torch.tensor
                            Returns a copy of the targets.
        target_depth      : torch.tensor
                            Returns a copy of the normalized quantized depth map.

        """
        divider = self.number_of_planes - 1
        if divider == 0:
            divider = 1
        return self.targets.detach().clone(), self.focus_target.detach().clone(), self.target_depth.detach().clone() / divider


    def set_targets(self):
        """
        Internal function for slicing the depth into planes without considering defocus. Users can query the results with get_targets() within the same class.
        """
        self.target_depth = self.target_depth * (self.number_of_planes - 1)
        self.target_depth = torch.round(self.target_depth, decimals = 0)
        self.targets      = torch.zeros(
                                        self.number_of_planes,
                                        self.target_image.shape[0],
                                        self.target_image.shape[1],
                                        self.target_image.shape[2],
                                        requires_grad = False,
                                        device = self.device
                                       )
        self.focus_target = torch.zeros_like(self.target_image, requires_grad = False)
        self.masks        = torch.zeros_like(self.targets)
        for i in range(self.number_of_planes):
            for ch in range(self.target_image.shape[0]):
                mask_zeros = torch.zeros_like(self.target_image[ch], dtype = torch.int)
                mask_ones = torch.ones_like(self.target_image[ch], dtype = torch.int)
                mask = torch.where(self.target_depth == i, mask_ones, mask_zeros)
                new_target = self.target_image[ch] * mask
                self.focus_target = self.focus_target + new_target.squeeze(0).squeeze(0).detach().clone()
                self.targets[i, ch] = new_target.squeeze(0).squeeze(0)
                self.masks[i, ch] = mask.detach().clone() 


    def add_defocus_blur(self):
        """
        Internal function for adding defocus blur to the multiplane targets. Users can query the results with get_targets() within the same class.
        """
        kernel_length = [self.target_blur_size, self.target_blur_size ]
        for ch in range(self.target_image.shape[0]):
            targets_cache = self.targets[:, ch].detach().clone()
            target = torch.sum(targets_cache, axis = 0)
            for i in range(self.number_of_planes):
                defocus = torch.zeros_like(targets_cache[i])
                for j in range(self.number_of_planes):
                    nsigma = [int(abs(i - j) * self.blur_ratio), int(abs(i -j) * self.blur_ratio)]
                    if torch.sum(targets_cache[j]) > 0:
                        if i == j:
                            nsigma = [0., 0.]
                        kernel = generate_2d_gaussian(kernel_length, nsigma).to(self.device)
                        kernel = kernel / torch.sum(kernel)
                        kernel = kernel.unsqueeze(0).unsqueeze(0)
                        target_current = target.detach().clone().unsqueeze(0).unsqueeze(0)
                        defocus_plane = torch.nn.functional.conv2d(target_current, kernel, padding = 'same')
                        defocus_plane = defocus_plane.view(defocus_plane.shape[-2], defocus_plane.shape[-1])
                        defocus = defocus + defocus_plane * torch.abs(self.masks[j, ch])
                self.targets[i, ch] = defocus
        self.targets = self.targets.detach().clone() * self.multiplier
    

    def __call__(self, image, target, plane_id = None):
        """
        Calculates the multiplane loss against a given target.
        
        Parameters
        ----------
        image         : torch.tensor
                        Image to compare with a target [3 x m x n].
        target        : torch.tensor
                        Target image for comparison [3 x m x n].
        plane_id      : int
                        Number of the plane under test.
        
        Returns
        -------
        loss          : torch.tensor
                        Computed loss.
        """
        l2 = self.weights[0] * self.loss_function(image, target)
        if isinstance(plane_id, type(None)):
            mask = self.masks
        else:
            mask= self.masks[plane_id, :]
        l2_mask = self.weights[1] * self.loss_function(image * mask, target * mask)
        l2_cor = self.weights[2] * self.loss_function(image * target, target * target)
        loss = l2 + l2_mask + l2_cor
        return loss


class perceptual_multiplane_loss():
    """
    Perceptual loss function for computing loss in multiplanar images. Unlike, previous methods, this loss function accounts for defocused parts of an image.
    """

    def __init__(self, target_image, target_depth, blur_ratio = 0.25, 
                 target_blur_size = 10, number_of_planes = 4, multiplier = 1., scheme = 'defocus', 
                 base_loss_weights = {'base_l2_loss': 1., 'loss_l2_mask': 1., 'loss_l2_cor': 1., 'base_l1_loss': 1., 'loss_l1_mask': 1., 'loss_l1_cor': 1.},
                 additional_loss_weights = {'cvvdp': 1.}, reduction = 'mean', return_components = False, device = torch.device('cpu')):
        """
        Parameters
        ----------
        target_image            : torch.tensor
                                    Color target image [3 x m x n].
        target_depth            : torch.tensor
                                    Monochrome target depth, same resolution as target_image.
        target_blur_size        : int
                                    Maximum target blur size.
        blur_ratio              : float
                                    Blur ratio, a value between zero and one.
        number_of_planes        : int
                                    Number of planes.
        multiplier              : float
                                    Multiplier to multipy with targets.
        scheme                  : str
                                    The type of the loss, `naive` without defocus or `defocus` with defocus.
        base_loss_weights       : list
                                    Weights of the base loss functions. Default is {'base_l2_loss': 1., 'loss_l2_mask': 1., 'loss_l2_cor': 1., 'base_l1_loss': 1., 'loss_l1_mask': 1., 'loss_l1_cor': 1.}.
        additional_loss_weights : dict
                                    Additional loss terms and their weights (e.g., {'cvvdp': 1.}). Supported loss terms are 'cvvdp', 'fvvdp', 'lpips', 'psnr', 'ssim', 'msssim'.
        reduction               : str
                                    Reduction can either be 'mean', 'none' or 'sum'. For more see: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
        return_components       : bool
                                    If True (False by default), returns the components of the loss as a dict.
        device                  : torch.device
                                    Device to be used (e.g., cuda, cpu, opencl).
        """
        self.device = device
        self.target_image     = target_image.float().to(self.device)
        self.target_depth     = target_depth.float().to(self.device)
        self.target_blur_size = target_blur_size
        if self.target_blur_size % 2 == 0:
            self.target_blur_size += 1
        self.number_of_planes = number_of_planes
        self.multiplier       = multiplier
        self.reduction        = reduction
        if self.reduction == 'none' and len(list(additional_loss_weights.keys())) > 0:
            logging.warning("Reduction cannot be 'none' for additional loss functions. Changing reduction to 'mean'.")
            self.reduction = 'mean'
        self.blur_ratio       = blur_ratio
        self.set_targets()
        if scheme == 'defocus':
            self.add_defocus_blur()
        self.base_loss_weights = base_loss_weights
        self.additional_loss_weights = additional_loss_weights
        self.return_components = return_components
        self.l1_loss_fn = torch.nn.L1Loss(reduction = self.reduction)
        self.l2_loss_fn = torch.nn.MSELoss(reduction = self.reduction)
        for key in self.additional_loss_weights.keys():
            if key == 'cvvdp':
                self.cvvdp = CVVDP()
            if key == 'fvvdp':
                self.fvvdp = FVVDP()
            if key == 'lpips':
                self.lpips = LPIPS()
            if key == 'psnr':
                self.psnr = PSNR()
            if key == 'ssim':
                self.ssim = SSIM()
            if key == 'msssim':
                self.msssim = MSSSIM()
        
    def get_targets(self):
        """
        Returns
        -------
        targets           : torch.tensor
                            Returns a copy of the targets.
        target_depth      : torch.tensor
                            Returns a copy of the normalized quantized depth map.

        """
        divider = self.number_of_planes - 1
        if divider == 0:
            divider = 1
        return self.targets.detach().clone(), self.focus_target.detach().clone(), self.target_depth.detach().clone() / divider


    def set_targets(self):
        """
        Internal function for slicing the depth into planes without considering defocus. Users can query the results with get_targets() within the same class.
        """
        self.target_depth = self.target_depth * (self.number_of_planes - 1)
        self.target_depth = torch.round(self.target_depth, decimals = 0)
        self.targets      = torch.zeros(
                                        self.number_of_planes,
                                        self.target_image.shape[0],
                                        self.target_image.shape[1],
                                        self.target_image.shape[2],
                                        requires_grad = False,
                                        device = self.device
                                       )
        self.focus_target = torch.zeros_like(self.target_image, requires_grad = False)
        self.masks        = torch.zeros_like(self.targets)
        for i in range(self.number_of_planes):
            for ch in range(self.target_image.shape[0]):
                mask_zeros = torch.zeros_like(self.target_image[ch], dtype = torch.int)
                mask_ones = torch.ones_like(self.target_image[ch], dtype = torch.int)
                mask = torch.where(self.target_depth == i, mask_ones, mask_zeros)
                new_target = self.target_image[ch] * mask
                self.focus_target = self.focus_target + new_target.squeeze(0).squeeze(0).detach().clone()
                self.targets[i, ch] = new_target.squeeze(0).squeeze(0)
                self.masks[i, ch] = mask.detach().clone() 


    def add_defocus_blur(self):
        """
        Internal function for adding defocus blur to the multiplane targets. Users can query the results with get_targets() within the same class.
        """
        kernel_length = [self.target_blur_size, self.target_blur_size ]
        for ch in range(self.target_image.shape[0]):
            targets_cache = self.targets[:, ch].detach().clone()
            target = torch.sum(targets_cache, axis = 0)
            for i in range(self.number_of_planes):
                defocus = torch.zeros_like(targets_cache[i])
                for j in range(self.number_of_planes):
                    nsigma = [int(abs(i - j) * self.blur_ratio), int(abs(i -j) * self.blur_ratio)]
                    if torch.sum(targets_cache[j]) > 0:
                        if i == j:
                            nsigma = [0., 0.]
                        kernel = generate_2d_gaussian(kernel_length, nsigma).to(self.device)
                        kernel = kernel / torch.sum(kernel)
                        kernel = kernel.unsqueeze(0).unsqueeze(0)
                        target_current = target.detach().clone().unsqueeze(0).unsqueeze(0)
                        defocus_plane = torch.nn.functional.conv2d(target_current, kernel, padding = 'same')
                        defocus_plane = defocus_plane.view(defocus_plane.shape[-2], defocus_plane.shape[-1])
                        defocus = defocus + defocus_plane * torch.abs(self.masks[j, ch])
                self.targets[i, ch] = defocus
        self.targets = self.targets.detach().clone() * self.multiplier
    

    def __call__(self, image, target, plane_id = None):
        """
        Calculates the multiplane loss against a given target.
        
        Parameters
        ----------
        image         : torch.tensor
                        Image to compare with a target [3 x m x n].
        target        : torch.tensor
                        Target image for comparison [3 x m x n].
        plane_id      : int
                        Number of the plane under test.
        
        Returns
        -------
        loss          : torch.tensor
                        Computed loss.
        """
        loss_components = {}
        if isinstance(plane_id, type(None)):
            mask = self.masks
        else:
            mask= self.masks[plane_id, :]
        l2 = self.base_loss_weights['base_l2_loss'] * self.l2_loss_fn(image, target)
        l2_mask = self.base_loss_weights['loss_l2_mask'] * self.l2_loss_fn(image * mask, target * mask)
        l2_cor = self.base_loss_weights['loss_l2_cor'] * self.l2_loss_fn(image * target, target * target)
        loss_components['l2'] = l2
        loss_components['l2_mask'] = l2_mask
        loss_components['l2_cor'] = l2_cor

        l1 = self.base_loss_weights['base_l1_loss'] * self.l1_loss_fn(image, target)
        l1_mask = self.base_loss_weights['loss_l1_mask'] * self.l1_loss_fn(image * mask, target * mask)
        l1_cor = self.base_loss_weights['loss_l1_cor'] * self.l1_loss_fn(image * target, target * target)
        loss_components['l1'] = l1
        loss_components['l1_mask'] = l1_mask
        loss_components['l1_cor'] = l1_cor

        for key in self.additional_loss_weights.keys():
            if key == 'cvvdp':
                loss_cvvdp = self.additional_loss_weights['cvvdp'] * self.cvvdp(image, target)
                loss_components['cvvdp'] = loss_cvvdp
            if key == 'fvvdp':
                loss_fvvdp = self.additional_loss_weights['fvvdp'] * self.fvvdp(image, target)
                loss_components['fvvdp'] = loss_fvvdp
            if key == 'lpips':
                loss_lpips = self.additional_loss_weights['lpips'] * self.lpips(image, target)
                loss_components['lpips'] = loss_lpips
            if key == 'psnr':
                loss_psnr = self.additional_loss_weights['psnr'] * self.psnr(image, target)
                loss_components['psnr'] = loss_psnr
            if key == 'ssim':
                loss_ssim = self.additional_loss_weights['ssim'] * self.ssim(image, target)
                loss_components['ssim'] = loss_ssim
            if key == 'msssim':
                loss_msssim = self.additional_loss_weights['msssim'] * self.msssim(image, target)
                loss_components['msssim'] = loss_msssim

        loss = torch.sum(torch.stack(list(loss_components.values())), dim = 0)

        if self.return_components:
            return loss, loss_components
        return loss
