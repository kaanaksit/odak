from odak.learn.perception.steerable_pyramid_filters import get_steerable_pyramid_filters
import torch
import numpy as np

from .spatial_steerable_pyramid import get_steerable_pyramid_filters

class SpatialSteerablePyramid():
    """
    This implements a real-valued steerable pyramid where the filtering is carried out spatially (using convolution)
    as opposed to multiplication in the Fourier domain.
    This has a number of optimisations over previous implementations that increase efficiency, but introduce some
    reconstruction error.
    use_bilinear_downup: This uses bilinear filtering when upsampling/downsampling, rather than the original approach
    of applying a large lowpass kernel and sampling even rows/columns
    filter_type: This can be used to select smaller filters than the original ones if desired.
        full: Original filter sizes
        cropped: Some filters are cut back in size by extracting the centre and scaling as appropriate.
        trained: Same as reduced, but the oriented kernels are replaced by learned 5x5 kernels.
    """

    def __init__(self, use_bilinear_downup=True, n_channels=1, 
        filter_size=9, n_orientations=6, filter_type="full",
        device=torch.device('cpu')):
        self.use_bilinear_downup = use_bilinear_downup
        self.device = device

        filters = get_steerable_pyramid_filters(filter_size, n_orientations, filter_type)
        def make_pad(filter):
            filter_size = filter.size(-1)
            pad_amt = (filter_size-1) // 2
            return torch.nn.ReflectionPad2d((pad_amt, pad_amt, pad_amt, pad_amt))

        if not self.use_bilinear_downup:
            self.filt_l = filters["l"].to(device)
            self.pad_l = make_pad(self.filt_l)
        self.filt_l0 = filters["l0"].to(device)
        self.pad_l0 = make_pad(self.filt_l0)
        self.filt_h0 = filters["h0"].to(device)
        self.pad_h0 = make_pad(self.filt_h0)
        for b in range(len(filters["b"])):
            filters["b"][b] = filters["b"][b].to(device)
        self.band_filters = filters["b"]
        self.pad_b = make_pad(self.band_filters[0])

        if n_channels != 1:
            def add_channels_to_filter(filter):
                padded = torch.zeros(n_channels, n_channels, filter.size()[2], filter.size()[3]).to(device)
                for channel in range(n_channels):
                    padded[channel, channel, :, :] = filter
                return padded
            self.filt_h0 = add_channels_to_filter(self.filt_h0)
            for b in range(len(self.band_filters)):
                self.band_filters[b] = add_channels_to_filter(self.band_filters[b])
            self.filt_l0 = add_channels_to_filter(self.filt_l0)
            if not self.use_bilinear_downup:
                self.filt_l = add_channels_to_filter(self.filt_l)

    def construct_pyramid(self, image, n_levels, multiple_highpass=False):
        pyramid = []

        # Make level 0, containing highpass, lowpass and the bands
        level0 = {}
        level0['h'] = torch.nn.functional.conv2d(self.pad_h0(image), self.filt_h0)
        #plt.imshow(level0['h'][0,0,...], cmap="gray", vmin=0, vmax=1)
        #plt.show()
        lowpass = torch.nn.functional.conv2d(self.pad_l0(image), self.filt_l0)
        level0['l'] = lowpass.clone()
        #np.save("lowpass_filtered.npy", level0['l'][0,...].permute(1,2,0).numpy())
        #quit()
        bands = []
        for filt_b in self.band_filters:
            bands.append(torch.nn.functional.conv2d(self.pad_b(lowpass), filt_b))
        level0['b'] = bands
        pyramid.append(level0)

        # Make intermediate levels
        for l in range(n_levels-2):
            level = {}
            if self.use_bilinear_downup:
                lowpass = torch.nn.functional.interpolate(lowpass, scale_factor=0.5, mode="area", recompute_scale_factor=False)
            else:
                lowpass = torch.nn.functional.conv2d(self.pad_l(lowpass), self.filt_l)
                lowpass = lowpass[:,:,::2,::2]
            level['l'] = lowpass.clone()
            bands = []
            for filt_b in self.band_filters:
                bands.append(torch.nn.functional.conv2d(self.pad_b(lowpass), filt_b))
            level['b'] = bands
            if multiple_highpass:
                #downsampled = torch.nn.functional.interpolate(image, scale_factor=0.5, mode="area")
                #level['h'] = torch.nn.functional.conv2d(self.pad_h0(downsampled), self.filt_h0)
                level['h'] = torch.nn.functional.conv2d(self.pad_h0(lowpass), self.filt_h0)
            pyramid.append(level)
        
        # Make final level (lowpass residual)
        level = {}
        if self.use_bilinear_downup:
            lowpass = torch.nn.functional.interpolate(lowpass, scale_factor=0.5, mode="area", recompute_scale_factor=False)
        else:
            lowpass = torch.nn.functional.conv2d(self.pad_l(lowpass), self.filt_l)
            lowpass = lowpass[:,:,::2,::2]
        level['l'] = lowpass
        pyramid.append(level)

        return pyramid

    def reconstruct_from_pyramid_upsample(self, pyramid):
        output_size = pyramid[0]['h'].size()[-2:]
        image = pyramid[-1]['l']
        image = torch.nn.functional.interpolate(image, size=output_size, mode="nearest", recompute_scale_factor=False)
        for level in reversed(pyramid[:-1]):
            for b in range(len(level['b'])):
                b_filtered = torch.nn.functional.conv2d(self.pad_b(level['b'][b]), -self.band_filters[b])
                image += torch.nn.functional.interpolate(b_filtered, size=output_size, mode="nearest", recompute_scale_factor=False)

        image = torch.nn.functional.conv2d(self.pad_l0(image), self.filt_l0)
        image += torch.nn.functional.conv2d(self.pad_h0(pyramid[0]['h']), self.filt_h0)
        
        return image

    def reconstruct_from_pyramid(self, pyramid):
        def upsample(image, size):
            if self.use_bilinear_downup:
                return torch.nn.functional.interpolate(image, size=size, mode="bilinear", align_corners=False, recompute_scale_factor=False)
            else:
                zeros = torch.zeros((image.size()[0], image.size()[1], image.size()[2]*2, image.size()[3]*2)).to(self.device)
                zeros[:,:,::2,::2] = image
                zeros = torch.nn.functional.conv2d(self.pad_l(zeros), self.filt_l)
                return zeros

        image = pyramid[-1]['l']
        for level in reversed(pyramid[:-1]):
            image = upsample(image, level['b'][0].size()[2:])
            for b in range(len(level['b'])):
                b_filtered = torch.nn.functional.conv2d(self.pad_b(level['b'][b]), -self.band_filters[b])
                image += b_filtered

        image = torch.nn.functional.conv2d(self.pad_l0(image), self.filt_l0)
        image += torch.nn.functional.conv2d(self.pad_h0(pyramid[0]['h']), self.filt_h0)
        
        return image