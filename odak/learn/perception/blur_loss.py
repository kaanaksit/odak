import torch

from .radially_varying_blur import RadiallyVaryingBlur

class BlurLoss():
    """ BlurLoss implements two of the losses presented in the paper.
        When blur_source is set to False, it implements blur_match from the paper, trying to match
        the input image to the blurred target image.
        When blur_source is set to True, it implements blur_lowpass from the paper, matching the blurred
        version of the input image to the blurred target image.
    =================================================================
    Parameters:
    alpha: parameter controlling foveation - larger values mean bigger pooling
        regions.
    real_image_width: The real width of the image as displayed to the user.
        Units don't matter as long as they are the same as for real_viewing_distance.
    real_viewing_distance: The real distance of the observer's eyes to the image plane.
        Units don't matter as long as they are the same as for real_image_width.
    mode: Foveation mode, either "quadratic" or "linear". Controls how pooling regions grow
        as you move away from the fovea. We got best results with "quadratic".
    =================================================================
    """
    def __init__(self, device=torch.device("cpu"),\
        alpha=0.08, real_image_width=0.2, real_viewing_distance=0.7, mode="quadratic",
        use_old_blur=False, blur_source=False):
        self.target = None
        self.device = device
        self.alpha = alpha
        self.real_image_width = real_image_width
        self.real_viewing_distance = real_viewing_distance
        self.mode = mode
        self.blur = None
        self.loss_func = torch.nn.MSELoss()
        self.use_old_blur = use_old_blur
        self.blur_source = blur_source

    def blur_image(self, image, gaze):
        if self.blur is None:
            self.blur = RadiallyVaryingBlur()
        return self.blur.blur(image, self.alpha, self.real_image_width, self.real_viewing_distance, gaze, self.mode)

    
    def __call__(self, image, target, gaze=[0.5,0.5]):
        blurred_target = self.blur_image(target, gaze)
        if self.blur_source:
            blurred_image = self.blur_image(image, gaze)
            return self.loss_func(blurred_image, blurred_target)
        else:
            return self.loss_func(image, blurred_target)
    
    def to(self, device):
        self.device = device
        return self

