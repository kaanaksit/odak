import torch

from .radially_varying_blur import RadiallyVaryingBlur
from .util import check_loss_inputs


class BlurLoss():
    """ 

    `BlurLoss` implements two different blur losses. When `blur_source` is set to `False`, it implements blur_match, trying to match the input image to the blurred target image. This tries to match the source input image to a blurred version of the target.

    When `blur_source` is set to `True`, it implements blur_lowpass, matching the blurred version of the input image to the blurred target image. This tries to match only the low frequencies of the source input image to the low frequencies of the target.

    The interface is similar to other `pytorch` loss functions, but note that the gaze location must be provided in addition to the source and target images.
    """


    def __init__(self, device=torch.device("cpu"),
                 alpha=0.2, real_image_width=0.2, real_viewing_distance=0.7, mode="quadratic", blur_source=False, equi=False):
        """
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
        mode                    : str 
                                    Foveation mode, either "quadratic" or "linear". Controls how pooling regions grow
                                    as you move away from the fovea. We got best results with "quadratic".
        blur_source             : bool
                                    If true, blurs the source image as well as the target before computing the loss.
        equi                    : bool
                                    If true, run the loss in equirectangular mode. The input is assumed to be an equirectangular
                                    format 360 image. The settings real_image_width and real_viewing distance are ignored.
                                    The gaze argument is instead interpreted as gaze angles, and should be in the range
                                    [-pi,pi]x[-pi/2,pi]
        """
        self.target = None
        self.device = device
        self.alpha = alpha
        self.real_image_width = real_image_width
        self.real_viewing_distance = real_viewing_distance
        self.mode = mode
        self.blur = None
        self.loss_func = torch.nn.MSELoss()
        self.blur_source = blur_source
        self.equi = equi

    def blur_image(self, image, gaze):
        if self.blur is None:
            self.blur = RadiallyVaryingBlur()
        return self.blur.blur(image, self.alpha, self.real_image_width, self.real_viewing_distance, gaze, self.mode, self.equi)

    def __call__(self, image, target, gaze=[0.5, 0.5]):
        """ 
        Calculates the Blur Loss.

        Parameters
        ----------
        image               : torch.tensor
                                Image to compute loss for. Should be an RGB image in NCHW format (4 dimensions)
        target              : torch.tensor
                                Ground truth target image to compute loss for. Should be an RGB image in NCHW format (4 dimensions)
        gaze                : list
                                Gaze location in the image, in normalized image coordinates (range [0, 1]) relative to the top left of the image.

        Returns
        -------

        loss                : torch.tensor
                                The computed loss.
        """
        check_loss_inputs("BlurLoss", image, target)
        blurred_target = self.blur_image(target, gaze)
        if self.blur_source:
            blurred_image = self.blur_image(image, gaze)
            return self.loss_func(blurred_image, blurred_target)
        else:
            return self.loss_func(image, blurred_target)

    def to(self, device):
        self.device = device
        return self
