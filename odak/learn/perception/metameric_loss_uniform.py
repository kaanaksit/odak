import torch

from .color_conversion import rgb_2_ycrcb, ycrcb_2_rgb
from .spatial_steerable_pyramid import SpatialSteerablePyramid, pad_image_for_pyramid
from .util import check_loss_inputs


def uniform_blur(image, pooling_size):
    original_size = image.size()[-2:]
    down = torch.nn.functional.interpolate(image, scale_factor=1. / pooling_size, mode="area")
    blurred = torch.nn.functional.interpolate(down, size=original_size, mode="bilinear")
    return blurred

class MetamericLossUniform():
    """
    Measures metameric loss between a given image and a metamer of the given target image.
    This variant of the metameric loss is not foveated - it applies uniform pooling sizes to the whole input image.
    """

    def __init__(self, device=torch.device('cpu'), pooling_size=32, n_pyramid_levels=5, n_orientations=2):
        """

        Parameters
        ----------
        pooling_size            : int
                                  Pooling size, in pixels. For example 32 will pool over 32x32 blocks of the image.
        n_pyramid_levels        : int 
                                  Number of levels of the steerable pyramid. Note that the image is padded
                                  so that both height and width are multiples of 2^(n_pyramid_levels), so setting this value
                                  too high will slow down the calculation a lot.
        n_orientations          : int 
                                  Number of orientations in the steerable pyramid. Can be 1, 2, 4 or 6.
                                  Increasing this will increase runtime.

        """
        self.target = None
        self.device = device
        self.pyramid_maker = None
        self.pooling_size = pooling_size
        self.n_pyramid_levels = n_pyramid_levels
        self.n_orientations = n_orientations

    def calc_statsmaps(self, image, pooling_size):

        if self.pyramid_maker is None or \
                self.pyramid_maker.device != self.device or \
                len(self.pyramid_maker.band_filters) != self.n_orientations or\
                self.pyramid_maker.filt_h0.size(0) != image.size(1):
            self.pyramid_maker = SpatialSteerablePyramid(
                use_bilinear_downup=False, n_channels=image.size(1),
                device=self.device, n_orientations=self.n_orientations, filter_type="cropped", filter_size=5)


        def find_stats(image_pyr_level, pooling_size):
            image_means = uniform_blur(image_pyr_level, pooling_size)
            image_meansq = uniform_blur(image_pyr_level*image_pyr_level, pooling_size)
            image_vars = image_meansq - (image_means*image_means)
            image_vars[image_vars < 1e-7] = 1e-7
            image_std = torch.sqrt(image_vars)
            if torch.any(torch.isnan(image_means)):
                print(image_means)
                raise Exception("NaN in image means!")
            if torch.any(torch.isnan(image_std)):
                print(image_std)
                raise Exception("NaN in image stdevs!")
            return image_means, image_std

        output_stats = []
        image_pyramid = self.pyramid_maker.construct_pyramid(
            image, self.n_pyramid_levels)
        curr_pooling_size = pooling_size
        means, variances = find_stats(image_pyramid[0]['h'], curr_pooling_size)
        output_stats.append(means)
        output_stats.append(variances)

        for l in range(0, len(image_pyramid)-1):
            for o in range(len(image_pyramid[l]['b'])):
                means, variances = find_stats(
                    image_pyramid[l]['b'][o], curr_pooling_size)
                output_stats.append(means)
                output_stats.append(variances)
            curr_pooling_size /= 2

        output_stats.append(image_pyramid[-1]["l"])
        return output_stats

    def metameric_loss_stats(self, statsmap_a, statsmap_b):
        loss = 0.0
        for a, b in zip(statsmap_a, statsmap_b):
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

    def __call__(self, image, target, image_colorspace="RGB", visualise_loss=False):
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
        visualise_loss      : bool
                                Shows a heatmap indicating which parts of the image contributed most to the loss. 

        Returns
        -------

        loss                : torch.tensor
                                The computed loss.
        """
        check_loss_inputs("MetamericLossUniform", image, target)
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
                self.target_stats = self.calc_statsmaps(self.target, self.pooling_size)
                self.target = target.detach().clone()
            image_stats = self.calc_statsmaps(image, self.pooling_size)

            if visualise_loss:
                self.visualise_loss_map(image_stats)
            loss = self.metameric_loss_stats(
                image_stats, self.target_stats)
            return loss
        else:
            raise Exception("Target of incorrect type")

    def gen_metamer(self, image):
        """ 
        Generates a metamer for an image, following the method in [this paper](https://dl.acm.org/doi/abs/10.1145/3450626.3459943)
        This function can be used on its own to generate a metamer for a desired image.

        Parameters
        ----------
        image   : torch.tensor
                  Image to compute metamer for. Should be an RGB image in NCHW format (4 dimensions)

        Returns
        -------
        metamer : torch.tensor
                  The generated metamer image
        """
        image = rgb_2_ycrcb(image)
        image_size = image.size()
        image = pad_image_for_pyramid(image, self.n_pyramid_levels)

        target_stats = self.calc_statsmaps(
            image, self.pooling_size)
        target_means = target_stats[::2]
        target_stdevs = target_stats[1::2]
        torch.manual_seed(0)
        noise_image = torch.rand_like(image)
        noise_pyramid = self.pyramid_maker.construct_pyramid(
            noise_image, self.n_pyramid_levels)
        input_pyramid = self.pyramid_maker.construct_pyramid(
            image, self.n_pyramid_levels)

        def match_level(input_level, target_mean, target_std):
            level = input_level.clone()
            level -= torch.mean(level)
            input_std = torch.sqrt(torch.mean(level * level))
            eps = 1e-6
            # Safeguard against divide by zero
            input_std[input_std < eps] = eps
            level /= input_std
            level *= target_std
            level += target_mean
            return level

        nbands = len(noise_pyramid[0]["b"])
        noise_pyramid[0]["h"] = match_level(
            noise_pyramid[0]["h"], target_means[0], target_stdevs[0])
        for l in range(len(noise_pyramid)-1):
            for b in range(nbands):
                noise_pyramid[l]["b"][b] = match_level(
                    noise_pyramid[l]["b"][b], target_means[1 + l * nbands + b], target_stdevs[1 + l * nbands + b])
        noise_pyramid[-1]["l"] = input_pyramid[-1]["l"]

        metamer = self.pyramid_maker.reconstruct_from_pyramid(
            noise_pyramid)
        metamer = ycrcb_2_rgb(metamer)
        # Crop to remove any padding
        metamer = metamer[:image_size[0], :image_size[1], :image_size[2], :image_size[3]]
        return metamer

    def to(self, device):
        self.device = device
        return self
