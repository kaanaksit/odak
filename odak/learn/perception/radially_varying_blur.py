import torch

from .foveation import make_pooling_size_map_lod, make_equi_pooling_size_map_lod


class RadiallyVaryingBlur():
    """ 

    The `RadiallyVaryingBlur` class provides a way to apply a radially varying blur to an image. Given a gaze location and information about the image and foveation, it applies a blur that will achieve the proper pooling size. The pooling size is chosen to appear the same at a range of display sizes and viewing distances, for a given `alpha` parameter value. For more information on how the pooling sizes are computed, please see [link coming soon]().

    The blur is accelerated by generating and sampling from MIP maps of the input image.

    This class caches the foveation information. This means that if it is run repeatedly with the same foveation parameters, gaze location and image size (e.g. in an optimisation loop) it won't recalculate the pooling maps.

    If you are repeatedly applying blur to images of different sizes (e.g. a pyramid) for best performance use one instance of this class per image size.

    """

    def __init__(self):
        self.lod_map = None
        self.equi = None

    def blur(self, image, alpha=0.2, real_image_width=0.2, real_viewing_distance=0.7, centre=None, mode="quadratic", equi=False):
        """
        Apply the radially varying blur to an image.

        Parameters
        ----------

        image                   : torch.tensor
                                    The image to blur, in NCHW format.
        alpha                   : float
                                    parameter controlling foveation - larger values mean bigger pooling regions.
        real_image_width        : float 
                                    The real width of the image as displayed to the user.
                                    Units don't matter as long as they are the same as for real_viewing_distance.
                                    Ignored in equirectangular mode (equi==True)
        real_viewing_distance   : float 
                                    The real distance of the observer's eyes to the image plane.
                                    Units don't matter as long as they are the same as for real_image_width.
                                    Ignored in equirectangular mode (equi==True)
        centre                  : tuple of floats
                                    The centre of the radially varying blur (the gaze location).
                                    Should be a tuple of floats containing normalised image coordinates in range [0,1]
                                    In equirectangular mode this should be yaw & pitch angles in [-pi,pi]x[-pi/2,pi/2]
        mode                    : str 
                                    Foveation mode, either "quadratic" or "linear". Controls how pooling regions grow
                                    as you move away from the fovea. We got best results with "quadratic".
        equi                    : bool
                                    If true, run the blur function in equirectangular mode. The input is assumed to be an equirectangular
                                    format 360 image. The settings real_image_width and real_viewing distance are ignored.
                                    The centre argument is instead interpreted as gaze angles, and should be in the range
                                    [-pi,pi]x[-pi/2,pi]

        Returns
        -------

        output                  : torch.tensor
                                    The blurred image
        """
        size = (image.size(-2), image.size(-1))

        # LOD map caching
        if self.lod_map is None or\
                self.size != size or\
                self.n_channels != image.size(1) or\
                self.alpha != alpha or\
                self.real_image_width != real_image_width or\
                self.real_viewing_distance != real_viewing_distance or\
                self.centre != centre or\
                self.mode != mode or\
                self.equi != equi:
            if not equi:
                self.lod_map = make_pooling_size_map_lod(
                    centre, (image.size(-2), image.size(-1)), alpha, real_image_width, real_viewing_distance, mode)
            else:
                self.lod_map = make_equi_pooling_size_map_lod(
                    centre, (image.size(-2), image.size(-1)), alpha, mode)
            self.size = size
            self.n_channels = image.size(1)
            self.alpha = alpha
            self.real_image_width = real_image_width
            self.real_viewing_distance = real_viewing_distance
            self.centre = centre
            self.lod_map = self.lod_map.to(image.device)
            self.lod_fraction = torch.fmod(self.lod_map, 1.0)
            self.lod_fraction = self.lod_fraction[None, None, ...].repeat(
                1, image.size(1), 1, 1)
            self.mode = mode
            self.equi = equi

        if self.lod_map.device != image.device:
            self.lod_map = self.lod_map.to(image.device)
        if self.lod_fraction.device != image.device:
            self.lod_fraction = self.lod_fraction.to(image.device)

        mipmap = [image]
        while mipmap[-1].size(-1) > 1 and mipmap[-1].size(-2) > 1:
            mipmap.append(torch.nn.functional.interpolate(
                mipmap[-1], scale_factor=0.5, mode="area", recompute_scale_factor=False))
        if mipmap[-1].size(-1) == 2:
            final_mip = torch.mean(mipmap[-1], axis=-1)[..., None]
            mipmap.append(final_mip)
        if mipmap[-1].size(-2) == 2:
            final_mip = torch.mean(mipmap[-2], axis=-2)[..., None, :]
            mipmap.append(final_mip)

        for l in range(len(mipmap)):
            if l == len(mipmap)-1:
                mipmap[l] = mipmap[l] * \
                    torch.ones(image.size(), device=image.device)
            else:
                for l2 in range(l-1, -1, -1):
                    mipmap[l] = torch.nn.functional.interpolate(mipmap[l], size=(
                        image.size(-2), image.size(-1)), mode="bilinear", align_corners=False, recompute_scale_factor=False)

        output = torch.zeros(image.size(), device=image.device)
        for l in range(len(mipmap)):
            if l == 0:
                mask = self.lod_map < (l+1)
            elif l == len(mipmap)-1:
                mask = self.lod_map >= l
            else:
                mask = torch.logical_and(
                    self.lod_map >= l, self.lod_map < (l+1))

            if l == len(mipmap)-1:
                blended_levels = mipmap[l]
            else:
                blended_levels = (1 - self.lod_fraction) * \
                    mipmap[l] + self.lod_fraction*mipmap[l+1]
            mask = mask[None, None, ...]
            mask = mask.repeat(1, image.size(1), 1, 1)
            output[mask] = blended_levels[mask]

        return output
