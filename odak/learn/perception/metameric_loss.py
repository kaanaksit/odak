import torch

from .color_conversion import rgb_2_ycrcb
from .spatial_steerable_pyramid import SpatialSteerablePyramid, pad_image_for_pyramid
from .radially_varying_blur import RadiallyVaryingBlur
from .foveation import make_radial_map
from .util import check_loss_inputs


class MetamericLoss():
    """
    The `MetamericLoss` class provides a perceptual loss function.

    Rather than exactly match the source image to the target, it tries to ensure the source is a *metamer* to the target image.

    Its interface is similar to other `pytorch` loss functions, but note that the gaze location must be provided in addition to the source and target images.
    """


    def __init__(self, device=torch.device('cpu'), alpha=0.2, real_image_width=0.2,
                 real_viewing_distance=0.7, n_pyramid_levels=5, mode="quadratic",
                 n_orientations=2, use_l2_foveal_loss=True, fovea_weight=20.0, use_radial_weight=False,
                 use_fullres_l0=False, equi=False):
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
        use_l2_foveal_loss      : bool 
                                    If true, for all the pixels that have pooling size 1 pixel in the 
                                    largest scale will use direct L2 against target rather than pooling over pyramid levels.
                                    In practice this gives better results when the loss is used for holography.
        fovea_weight            : float 
                                    A weight to apply to the foveal region if use_l2_foveal_loss is set to True.
        use_radial_weight       : bool 
                                    If True, will apply a radial weighting when calculating the difference between
                                    the source and target stats maps. This weights stats closer to the fovea more than those
                                    further away.
        use_fullres_l0          : bool 
                                    If true, stats for the lowpass residual are replaced with blurred versions
                                    of the full-resolution source and target images.
        equi                    : bool
                                    If true, run the loss in equirectangular mode. The input is assumed to be an equirectangular
                                    format 360 image. The settings real_image_width and real_viewing distance are ignored.
                                    The gaze argument is instead interpreted as gaze angles, and should be in the range
                                    [-pi,pi]x[-pi/2,pi]
        """
        self.target = None
        self.device = device
        self.pyramid_maker = None
        self.alpha = alpha
        self.real_image_width = real_image_width
        self.real_viewing_distance = real_viewing_distance
        self.blurs = None
        self.n_pyramid_levels = n_pyramid_levels
        self.n_orientations = n_orientations
        self.mode = mode
        self.use_l2_foveal_loss = use_l2_foveal_loss
        self.fovea_weight = fovea_weight
        self.use_radial_weight = use_radial_weight
        self.use_fullres_l0 = use_fullres_l0
        self.equi = equi
        if self.use_fullres_l0 and self.use_l2_foveal_loss:
            raise Exception(
                "Can't use use_fullres_l0 and use_l2_foveal_loss options together in MetamericLoss!")

    def calc_statsmaps(self, image, gaze=None, alpha=0.01, real_image_width=0.3,
                       real_viewing_distance=0.6, mode="quadratic", equi=False):

        if self.pyramid_maker is None or \
                self.pyramid_maker.device != self.device or \
                len(self.pyramid_maker.band_filters) != self.n_orientations or\
                self.pyramid_maker.filt_h0.size(0) != image.size(1):
            self.pyramid_maker = SpatialSteerablePyramid(
                use_bilinear_downup=False, n_channels=image.size(1),
                device=self.device, n_orientations=self.n_orientations, filter_type="cropped", filter_size=5)

        if self.blurs is None or len(self.blurs) != self.n_pyramid_levels:
            self.blurs = [RadiallyVaryingBlur()
                          for i in range(self.n_pyramid_levels)]

        def find_stats(image_pyr_level, blur):
            image_means = blur.blur(
                image_pyr_level, alpha, real_image_width, real_viewing_distance, centre=gaze, mode=mode, equi=self.equi)
            image_meansq = blur.blur(image_pyr_level*image_pyr_level, alpha,
                                     real_image_width, real_viewing_distance, centre=gaze, mode=mode, equi=self.equi)

            image_vars = image_meansq - (image_means*image_means)
            image_vars[image_vars < 1e-7] = 1e-7
            image_std = torch.sqrt(image_vars)
            if torch.any(torch.isnan(image_means)):
                print(image_means)
                raise Exception("NaN in image means!")
            if torch.any(torch.isnan(image_std)):
                print(image_std)
                raise Exception("NaN in image stdevs!")
            if self.use_fullres_l0:
                mask = blur.lod_map > 1e-6
                mask = mask[None, None, ...]
                if image_means.size(1) > 1:
                    mask = mask.repeat(1, image_means.size(1), 1, 1)
                matte = torch.zeros_like(image_means)
                matte[mask] = 1.0
                return image_means * matte, image_std * matte
            return image_means, image_std
        output_stats = []
        image_pyramid = self.pyramid_maker.construct_pyramid(
            image, self.n_pyramid_levels)
        means, variances = find_stats(image_pyramid[0]['h'], self.blurs[0])
        if self.use_l2_foveal_loss:
            self.fovea_mask = torch.zeros(image.size(), device=image.device)
            for i in range(self.fovea_mask.size(1)):
                self.fovea_mask[0, i, ...] = 1.0 - \
                    (self.blurs[0].lod_map / torch.max(self.blurs[0].lod_map))
                self.fovea_mask[0, i, self.blurs[0].lod_map < 1e-6] = 1.0
            self.fovea_mask = torch.pow(self.fovea_mask, 10.0)
            #self.fovea_mask     = torch.nn.functional.interpolate(self.fovea_mask, scale_factor=0.125, mode="area")
            #self.fovea_mask     = torch.nn.functional.interpolate(self.fovea_mask, size=(image.size(-2), image.size(-1)), mode="bilinear")
            periphery_mask = 1.0 - self.fovea_mask
            self.periphery_mask = periphery_mask.clone()
            output_stats.append(means * periphery_mask)
            output_stats.append(variances * periphery_mask)
        else:
            output_stats.append(means)
            output_stats.append(variances)

        for l in range(0, len(image_pyramid)-1):
            for o in range(len(image_pyramid[l]['b'])):
                means, variances = find_stats(
                    image_pyramid[l]['b'][o], self.blurs[l])
                if self.use_l2_foveal_loss:
                    output_stats.append(means * periphery_mask)
                    output_stats.append(variances * periphery_mask)
                else:
                    output_stats.append(means)
                    output_stats.append(variances)
            if self.use_l2_foveal_loss:
                periphery_mask = torch.nn.functional.interpolate(
                    periphery_mask, scale_factor=0.5, mode="area", recompute_scale_factor=False)

        if self.use_l2_foveal_loss:
            output_stats.append(image_pyramid[-1]["l"] * periphery_mask)
        elif self.use_fullres_l0:
            output_stats.append(self.blurs[0].blur(
                image, alpha, real_image_width, real_viewing_distance, gaze, mode))
        else:
            output_stats.append(image_pyramid[-1]["l"])
        return output_stats

    def metameric_loss_stats(self, statsmap_a, statsmap_b, gaze):
        loss = 0.0
        for a, b in zip(statsmap_a, statsmap_b):
            if self.use_radial_weight:
                radii = make_radial_map(
                    [a.size(-2), a.size(-1)], gaze).to(a.device)
                weights = 1.1 - (radii * radii * radii * radii)
                weights = weights[None, None, ...].repeat(1, a.size(1), 1, 1)
                loss += torch.nn.MSELoss()(weights*a, weights*b)
            else:
                loss += torch.nn.MSELoss()(a, b)
        loss /= len(statsmap_a)
        return loss

    def visualise_loss_map(self, image_stats):
        loss_map = torch.zeros(image_stats[0].size()[-2:])
        for i in range(len(image_stats)):
            stats = image_stats[i]
            target_stats = self.target_stats[i]
            stat_mse_map = torch.sqrt(torch.pow(stats - target_stats, 2))
            stat_mse_map = torch.nn.functional.interpolate(stat_mse_map, size=loss_map.size(
            ), mode="bilinear", align_corners=False, recompute_scale_factor=False)
            loss_map += stat_mse_map[0, 0, ...]
        self.loss_map = loss_map

    def __call__(self, image, target, gaze=[0.5, 0.5], image_colorspace="RGB", visualise_loss=False):
        """ 
        Calculates the Metameric Loss.

        Parameters
        ----------
        image               : torch.tensor
                                Image to compute loss for. Should be an RGB image in NCHW format (4 dimensions)
        target              : torch.tensor
                                Ground truth target image to compute loss for. Should be an RGB image in NCHW format (4 dimensions)
        image_colorspace    : str
                                The current colorspace of your image and target. Ignored if input does not have 3 channels.
                                accepted values: RGB, YCrCb.
        gaze                : list
                                Gaze location in the image, in normalized image coordinates (range [0, 1]) relative to the top left of the image.
        visualise_loss      : bool
                                Shows a heatmap indicating which parts of the image contributed most to the loss. 

        Returns
        -------

        loss                : torch.tensor
                                The computed loss.
        """
        check_loss_inputs("MetamericLoss", image, target)
        # Pad image and target if necessary
        image = pad_image_for_pyramid(image, self.n_pyramid_levels)
        target = pad_image_for_pyramid(target, self.n_pyramid_levels)
        # If input is RGB, convert to YCrCb.
        if image.size(1) == 3 and image_colorspace == "RGB":
            image = rgb_2_ycrcb(image)
            target = rgb_2_ycrcb(target)
        if self.target is None:
            self.target = torch.zeros(target.shape).to(target.device)
        if type(target) == type(self.target):
            if not torch.all(torch.eq(target, self.target)):
                self.target = target.detach().clone()
                self.target_stats = self.calc_statsmaps(
                    self.target,
                    gaze=gaze,
                    alpha=self.alpha,
                    real_image_width=self.real_image_width,
                    real_viewing_distance=self.real_viewing_distance,
                    mode=self.mode
                )
                self.target = target.detach().clone()
            image_stats = self.calc_statsmaps(
                image,
                gaze=gaze,
                alpha=self.alpha,
                real_image_width=self.real_image_width,
                real_viewing_distance=self.real_viewing_distance,
                mode=self.mode
            )
            if visualise_loss:
                self.visualise_loss_map(image_stats)
            if self.use_l2_foveal_loss:
                peripheral_loss = self.metameric_loss_stats(
                    image_stats, self.target_stats, gaze)
                foveal_loss = torch.nn.MSELoss()(self.fovea_mask*image, self.fovea_mask*target)
                # New weighting - evenly weight fovea and periphery.
                loss = peripheral_loss + self.fovea_weight * foveal_loss
            else:
                loss = self.metameric_loss_stats(
                    image_stats, self.target_stats, gaze)
            return loss
        else:
            raise Exception("Target of incorrect type")

    def to(self, device):
        self.device = device
        return self
