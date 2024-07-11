import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from torch.autograd import Variable
from ..tools import blur_gaussian, generate_2d_gaussian, zero_pad, crop_center


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
                 target_blur_size = 10, number_of_planes = 4, multiplier = 1., scheme = 'defocus', 
                 base_loss_fn = 'l2', additional_loss_terms = ['cvvdp'], base_loss_weights = {'base_loss': 1., 'loss_mask': 1., 'loss_cor': 1.},
                 additional_loss_weights = {}, reduction = 'mean', return_components = False, device = torch.device('cpu')):
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
        base_loss_fn            : str
                                    Base loss function, 'l2' by default. Options are ['l2', 'l1'].
        additional_loss_terms   : list
                                    Additional loss terms, ['cvvdp'] by default. Options are ['psnr', 'ssim', 'ms-ssim', 'lpips', 'fvvdp', 'cvvdp'].
        base_loss_weights       : list
                                    Weights of the base loss function.
        additional_loss_weights : dict
                                    Additional loss weights (e.g. {'cvvdp': 1.}). For not specified terms, the weight is 1.
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
        if reduction == 'none' and 'cvvdp' in additional_loss_terms:
            raise ValueError('Reduction cannot be none for cvvdp, please set reduction to mean or sum')
        if reduction == 'none' and 'fvvdp' in additional_loss_terms:
            raise ValueError('Reduction cannot be none for fvvdp, please set reduction to mean or sum')
        self.return_components = return_components
        self.blur_ratio       = blur_ratio
        self.set_targets()
        if scheme == 'defocus':
            self.add_defocus_blur()
        if base_loss_fn == 'l2':
            self.loss_function = torch.nn.MSELoss(reduction = self.reduction)
        elif base_loss_fn == 'l1':
            self.loss_function = torch.nn.L1Loss(reduction = self.reduction)
        else:
            raise ValueError('Base loss function must be either l1 or l2')
        self.additional_loss_terms = additional_loss_terms
        self.additional_loss_weights = additional_loss_weights
        self.base_loss_weights = base_loss_weights
        if 'cvvdp' in additional_loss_terms:
            try:

                import pycvvdp
                self.cvvdp = pycvvdp.cvvdp(display_name='standard_4k', device=self.device)
            except:
                logging.warning('ColorVideoVDP is missing, consider installing by visiting: https://github.com/gfxdisp/ColorVideoVDP')
        if 'fvvdp' in additional_loss_terms:
            try:
                import pyfvvdp
                self.fvvdp = pyfvvdp.fvvdp(display_name='standard_4k', heatmap='none', device=self.device)
            except:
                logging.warning('FovVideoVDP is missing')
        if 'lpips' in additional_loss_terms:
            try:
                import torchmetrics
                self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
            except:
                logging.warning('torchmetrics is missing, consider installing by visiting: https://pypi.org/project/torchmetrics')


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
                sigmas = torch.linspace(start = 0, end = self.target_blur_size, steps = self.number_of_planes)
                sigmas = sigmas - i * self.target_blur_size / (self.number_of_planes - 1 + 1e-10)
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
        components = {}
        base_loss = self.base_loss_weights['base_loss'] * self.loss_function(image, target)
        if isinstance(plane_id, type(None)):
            mask = self.masks
        else:
            mask= self.masks[plane_id, :]
        base_loss_mask = self.base_loss_weights['loss_mask'] * self.loss_function(image * mask, target * mask)
        base_loss_cor = self.base_loss_weights['loss_cor'] * self.loss_function(image * target, target * target)
        loss = base_loss + base_loss_mask + base_loss_cor
        components['base_loss'] = base_loss
        if 'cvvdp' in self.additional_loss_terms:
            image_clamped = torch.clamp(image, 0., 1.)
            l_ColorVideoVDP = self.cvvdp.loss(image_clamped, target, dim_order = 'CHW')
            if 'cvvdp' in self.additional_loss_weights.keys():
                l_ColorVideoVDP = l_ColorVideoVDP * self.additional_loss_weights['cvvdp']
                loss += l_ColorVideoVDP
            else:
                loss += l_ColorVideoVDP
            components['cvvdp'] = l_ColorVideoVDP
        if 'fvvdp' in self.additional_loss_terms:
            image_clamped = torch.clamp(image, 0., 1.)
            l_FovVideoVDP = self.fvvdp.predict(image_clamped, target, dim_order = 'CHW')[0]
            if 'fvvdp' in self.additional_loss_weights.keys():
                l_FovVideoVDP = l_FovVideoVDP * self.additional_loss_weights['fvvdp']
                loss += l_FovVideoVDP
            else:
                loss += l_FovVideoVDP
            components['fvvdp'] = l_FovVideoVDP
        if 'psnr' in self.additional_loss_terms:
            from torchmetrics.functional.image import peak_signal_noise_ratio
            l_PSNR = peak_signal_noise_ratio(image, target)
            if 'psnr' in self.additional_loss_weights.keys():
                l_PSNR = l_PSNR * self.additional_loss_weights['psnr']
                loss += l_PSNR
            else:
                loss += l_PSNR
            components['psnr'] = l_PSNR
        if 'ssim' in self.additional_loss_terms:
            from torchmetrics.functional.image import structural_similarity_index_measure
            l_SSIM = structural_similarity_index_measure(image.unsqueeze(0), target.unsqueeze(0))
            if 'ssim' in self.additional_loss_weights.keys():
                l_SSIM = l_SSIM * self.additional_loss_weights['ssim']
                loss += l_SSIM
            else:
                loss += l_SSIM
            components['ssim'] = l_SSIM
        if 'ms-ssim' in self.additional_loss_terms:
            from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
            l_MSSSIM = multiscale_structural_similarity_index_measure(image.unsqueeze(0), target.unsqueeze(0), data_range=1.0)
            if 'ms-ssim' in self.additional_loss_weights.keys():
                l_MSSSIM = l_MSSSIM * self.additional_loss_weights['ms-ssim']
                loss += l_MSSSIM
            else:
                loss += l_MSSSIM
            components['ms-ssim'] = l_MSSSIM
        if 'lpips' in self.additional_loss_terms:
            lpips_image = image
            lpips_target = target
            if lpips_image.shape[0] == 1:
                lpips_image = lpips_image.repeat(3, 1, 1)
                lpips_target = lpips_target.repeat(3, 1, 1)
            if len(lpips_image.shape) == 3:
                lpips_image = lpips_image.unsqueeze(0)
                lpips_target = lpips_target.unsqueeze(0)
            lpips_image = (lpips_image * 2 - 1).clamp(-1, 1)
            lpips_target = (lpips_target * 2 - 1).clamp(-1, 1)
            l_LPIPS = self.lpips(lpips_image, lpips_target)
            if 'lpips' in self.additional_loss_weights.keys():
                l_LPIPS = l_LPIPS * self.additional_loss_weights['lpips']
                loss += l_LPIPS
            else:
                loss += l_LPIPS
            components['lpips'] = l_LPIPS
        if self.return_components:
            return loss, components
        else:
            print('Components: ', components)
            return loss
