import torch
import math

from .metameric_loss import MetamericLoss
from .color_conversion import ycrcb_2_rgb, rgb_2_ycrcb

class MetamerMSELoss():
    """ 
    Measures the MSE between a given image and a metamer of the given target image.

    Parameters
    ----------
    alpha                   : float
                                parameter controlling foveation - larger values mean bigger pooling regions.
    real_image_width        : float 
                                The real width of the image as displayed to the user.
                                Units don't matter as long as they are the same as for real_viewing_distance.
    real_viewing_distance   : float 
                                The real distance of the observer's eyes to the image plane.
                                Units don't matter as long as they are the same as for real_image_width.
    n_pyramid_levels        : int 
                                Number of levels of the steerable pyramid. Note that the image is padded
                                so that both height and width are multiples of 2^(n_pyramid_levels), so setting this value
                                too high will slow down the calculation a lot.
    mode                    : str 
                                Foveation mode, either "quadratic" or "linear". Controls how pooling regions grow
                                as you move away from the fovea. We got best results with "quadratic".
    n_orientations          : int 
                                Number of orientations in the steerable pyramid. Can be 1, 2, 4 or 6.
                                Increasing this will increase runtime.
    """
    def __init__(self, device=torch.device("cpu"),\
        alpha=0.08, real_image_width=0.2, real_viewing_distance=0.7, mode="quadratic",
        n_pyramid_levels=5, n_orientations=2):
        self.target = None
        self.target_metamer = None
        self.metameric_loss = MetamericLoss(device=device, alpha=alpha, real_image_width=real_image_width,\
            real_viewing_distance=real_viewing_distance, 
            n_pyramid_levels=n_pyramid_levels, n_orientations=n_orientations, use_l2_foveal_loss=False)
        self.loss_func = torch.nn.MSELoss()

    def gen_metamer(self, image, gaze):
        """ 
        Generates a metamer for an image, following the method in [this paper](https://dl.acm.org/doi/abs/10.1145/3450626.3459943)
        This function can be used on its own to generate a metamer for a desired image.

        Parameters
        ----------
        image   : torch.tensor
                Image to compute metamer for. Should be an RGB image in NCHW format (4 dimensions)
        gaze    : list
                Gaze location in the image, in normalized image coordinates (range [0, 1]) relative to the top left of the image.
        
        Returns
        =======

        metamer : torch.tensor
                The generated metamer image
        """
        image = rgb_2_ycrcb(image)
        target_stats = self.metameric_loss.calc_statsmaps(image, gaze=gaze, alpha=self.metameric_loss.alpha)        
        target_means = target_stats[::2]
        target_stdevs = target_stats[1::2]
        torch.manual_seed(0)
        noise_image = torch.rand_like(image)
        noise_pyramid = self.metameric_loss.pyramid_maker.construct_pyramid(noise_image, self.metameric_loss.n_pyramid_levels)
        input_pyramid = self.metameric_loss.pyramid_maker.construct_pyramid(image, self.metameric_loss.n_pyramid_levels)

        def match_level(input_level, target_mean, target_std):
            level = input_level.clone()
            level -= torch.mean(level)
            input_std = torch.sqrt(torch.mean(level * level))
            eps = 1e-6
            input_std[input_std < eps] = eps #Safeguard against divide by zero
            level /= input_std
            level *= target_std
            level += target_mean
            return level

        nbands = len(noise_pyramid[0]["b"])
        noise_pyramid[0]["h"] = match_level(noise_pyramid[0]["h"], target_means[0], target_stdevs[0])
        for l in range(len(noise_pyramid)-1):
            for b in range(nbands):
                noise_pyramid[l]["b"][b] = match_level(noise_pyramid[l]["b"][b], target_means[1 + l * nbands + b], target_stdevs[1 + l * nbands + b])
        noise_pyramid[-1]["l"] = input_pyramid[-1]["l"]

        metamer = self.metameric_loss.pyramid_maker.reconstruct_from_pyramid(noise_pyramid)
        metamer = ycrcb_2_rgb(metamer)
        return metamer

    
    def __call__(self, image, target, gaze=[0.5,0.5]):
        """ 
        Calculates the Metamer MSE Loss.

        Parameters
        ----------
        image   : torch.tensor
                Image to compute loss for. Should be an RGB image in NCHW format (4 dimensions)
        target  : torch.tensor
                Ground truth target image to compute loss for. Should be an RGB image in NCHW format (4 dimensions)
        gaze    : list
                Gaze location in the image, in normalized image coordinates (range [0, 1]) relative to the top left of the image.

        Returns
        =======

        loss                : torch.tensor
                                The computed loss.
        """
        # Pad image and target if necessary
        min_divisor = 2 ** self.metameric_loss.n_pyramid_levels
        height = image.size(2)
        width = image.size(3)
        required_height = math.ceil(height / min_divisor) * min_divisor
        required_width = math.ceil(width / min_divisor) * min_divisor
        if required_height > height or required_width > width:
            # We need to pad!
            pad = torch.nn.ReflectionPad2d((0,0,required_height-height, required_width-width))
            image = pad(image)
            target = pad(target)

        if target is not self.target or self.target is None:
            self.target_metamer = self.gen_metamer(target, gaze)
            self.target = target
        
        return self.loss_func(image, self.target_metamer)
    
    def to(self, device):
        self.metameric_loss = self.metameric_loss.to(device)
        return self
