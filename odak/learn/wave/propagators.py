import torch
import numpy as np
import logging
from .classical import get_propagation_kernel, custom
from .util import wavenumber, generate_complex_field, calculate_amplitude, calculate_phase
from ..tools import zero_pad, crop_center, circular_binary_mask


class propagator():
    """
    A light propagation model that propagates light to desired image plane with two separate propagations. 
    We use this class in our various works including `Kavaklı et al., Realistic Defocus Blur for Multiplane Computer-Generated Holography`.
    """
    def __init__(
                 self,
                 resolution = [1920, 1080],
                 wavelengths = [515e-9,],
                 pixel_pitch = 8e-6,
                 resolution_factor = 1,
                 number_of_frames = 1,
                 number_of_depth_layers = 1,
                 volume_depth = 1e-2,
                 image_location_offset = 5e-3,
                 propagation_type = 'Bandlimited Angular Spectrum',
                 propagator_type = 'back and forth',
                 back_and_forth_distance = 0.3,
                 laser_channel_power = None,
                 aperture = None,
                 aperture_size = None,
                 distances = None,
                 aperture_samples = [20, 20, 5, 5],
                 method = 'conventional',
                 device = torch.device('cpu')
                ):
        """
        Parameters
        ----------
        resolution              : list
                                  Resolution.
        wavelengths             : float
                                  Wavelength of light in meters.
        pixel_pitch             : float
                                  Pixel pitch in meters.
        resolution_factor       : int
                                  Resolution factor for scaled simulations.
        number_of_frames        : int
                                  Number of hologram frames.
                                  Typically, there are three frames, each one for a single color primary.
        number_of_depth_layers  : int
                                  Equ-distance number of depth layers within the desired volume. If `distances` parameter is passed, this value will be automatically set to the length of the `distances` verson provided.
        volume_depth            : float
                                  Width of the volume along the propagation direction.
        image_location_offset   : float
                                  Center of the volume along the propagation direction.
        propagation_type        : str
                                  Propagation type. 
                                  See ropagate_beam() and odak.learn.wave.get_propagation_kernel() for more.
        propagator_type         : str
                                  Propagator type.
                                  The options are `back and forth` and `forward` propagators.
        back_and_forth_distance : float
                                  Zero mode distance for `back and forth` propagator type.
        laser_channel_power     : torch.tensor
                                  Laser channel powers for given number of frames and number of wavelengths.
        aperture                : torch.tensor
                                  Aperture at the Fourier plane.
        aperture_size           : float
                                  Aperture width for a circular aperture.
        aperture_samples        : list
                                  When using `Impulse Response Fresnel` propagation, these sample counts along X and Y will be used to represent a rectangular aperture. First two is for hologram plane pixel and the last two is for image plane pixel.
        distances               : torch.tensor
                                  Propagation distances in meters.
        method                  : str
                                  Hologram type conventional or multi-color.
        device                  : torch.device
                                  Device to be used for computation. For more see torch.device().
        """
        self.device = device
        self.pixel_pitch = pixel_pitch
        self.wavelengths = wavelengths
        self.resolution = resolution
        self.propagation_type = propagation_type
        if self.propagation_type != 'Impulse Response Fresnel':
            resolution_factor = 1
        self.resolution_factor = resolution_factor
        self.number_of_frames = number_of_frames
        self.number_of_depth_layers = number_of_depth_layers
        self.number_of_channels = len(self.wavelengths)
        self.volume_depth = volume_depth
        self.image_location_offset = image_location_offset
        self.propagator_type = propagator_type
        self.aperture_samples = aperture_samples
        self.zero_mode_distance = torch.tensor(back_and_forth_distance, device = device)
        self.method = method
        self.aperture = aperture
        self.init_distances(distances)
        self.init_kernels()
        self.init_channel_power(laser_channel_power)
        self.init_phase_scale()
        self.set_aperture(aperture, aperture_size)


    def init_distances(self, distances):
        """
        Internal function to initialize distances.

        Parameters
        ----------
        distances               : torch.tensor
                                  Propagation distances.
        """
        if isinstance(distances, type(None)):
            self.distances = torch.linspace(-self.volume_depth / 2., self.volume_depth / 2., self.number_of_depth_layers) + self.image_location_offset
        else:
            self.distances = torch.as_tensor(distances)
            self.number_of_depth_layers = self.distances.shape[0]
        logging.warning('Distances: {}'.format(self.distances))


    def init_kernels(self):
        """
        Internal function to initialize kernels.
        """
        self.generated_kernels = torch.zeros(
                                             self.number_of_depth_layers,
                                             self.number_of_channels,
                                             device = self.device
                                            )
        self.kernels = torch.zeros(
                                   self.number_of_depth_layers,
                                   self.number_of_channels,
                                   self.resolution[0] * self.resolution_factor * 2,
                                   self.resolution[1] * self.resolution_factor * 2,
                                   dtype = torch.complex64,
                                   device = self.device
                                  )


    def init_channel_power(self, channel_power):
        """
        Internal function to set the starting phase of the phase-only hologram.
        """
        self.channel_power = channel_power
        if isinstance(self.channel_power, type(None)):
            self.channel_power = torch.eye(
                                           self.number_of_frames,
                                           self.number_of_channels,
                                           device = self.device,
                                           requires_grad = False
                                          )


    def init_phase_scale(self):
        """
        Internal function to set the phase scale.
        In some cases, you may want to modify this init to ratio phases for different color primaries as an SLM is configured for a specific central wavelength.
        """
        self.phase_scale = torch.tensor(
                                        [
                                         1.,
                                         1.,
                                         1.
                                        ],
                                        requires_grad = False,
                                        device = self.device
                                       )


    def set_aperture(self, aperture = None, aperture_size = None):
        """
        Set aperture in the Fourier plane.


        Parameters
        ----------
        aperture        : torch.tensor
                          Aperture at the original resolution of a hologram.
                          If aperture is provided as None, it will assign a circular aperture at the size of the short edge (width or height).
        aperture_size   : int
                          If no aperture is provided, this will determine the size of the circular aperture.
        """
        if isinstance(aperture, type(None)):
            if isinstance(aperture_size, type(None)):
                aperture_size = torch.max(
                                          torch.tensor([
                                                        self.resolution[0] * self.resolution_factor, 
                                                        self.resolution[1] * self.resolution_factor
                                                       ])
                                         )
            self.aperture = circular_binary_mask(
                                                 self.resolution[0] * self.resolution_factor * 2,
                                                 self.resolution[1] * self.resolution_factor * 2,
                                                 aperture_size,
                                                ).to(self.device) * 1.
        else:
            self.aperture = zero_pad(aperture).to(self.device) * 1.


    def get_laser_powers(self):
        """
        Internal function to get the laser powers.

        Returns
        -------
        laser_power      : torch.tensor
                           Laser powers.
        """
        if self.method == 'conventional':
            laser_power = self.channel_power
        if self.method == 'multi-color':
            laser_power = torch.abs(torch.cos(self.channel_power))
        return laser_power


    def set_laser_powers(self, laser_power):
        """
        Internal function to set the laser powers.

        Parameters
        -------
        laser_power      : torch.tensor
                           Laser powers.
        """
        self.channel_power = laser_power



    def get_kernels(self):
        """
        Function to return the kernels used in the light transport.
        
        Returns
        -------
        kernels           : torch.tensor
                            Kernel amplitudes.
        """
        h = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(self.kernels)))
        kernels_amplitude = calculate_amplitude(h)
        kernels_phase = calculate_phase(h)
        return kernels_amplitude, kernels_phase


    def __call__(self, input_field, channel_id, depth_id):
        """
        Function that represents the forward model in hologram optimization.

        Parameters
        ----------
        input_field         : torch.tensor
                              Input complex input field.
        channel_id          : int
                              Identifying the color primary to be used.
        depth_id            : int
                              Identifying the depth layer to be used.

        Returns
        -------
        output_field        : torch.tensor
                              Propagated output complex field.
        """
        distance = self.distances[depth_id]
        if not self.generated_kernels[depth_id, channel_id]:
            if self.propagator_type == 'forward':
                H = get_propagation_kernel(
                                           nu = self.resolution[0] * 2,
                                           nv = self.resolution[1] * 2,
                                           dx = self.pixel_pitch,
                                           wavelength = self.wavelengths[channel_id],
                                           distance = distance,
                                           device = self.device,
                                           propagation_type = self.propagation_type,
                                           samples = self.aperture_samples,
                                           scale = self.resolution_factor
                                          )
            elif self.propagator_type == 'back and forth':
                H_forward = get_propagation_kernel(
                                                   nu = self.resolution[0] * 2,
                                                   nv = self.resolution[1] * 2,
                                                   dx = self.pixel_pitch,
                                                   wavelength = self.wavelengths[channel_id],
                                                   distance = self.zero_mode_distance,
                                                   device = self.device,
                                                   propagation_type = self.propagation_type,
                                                   samples = self.aperture_samples,
                                                   scale = self.resolution_factor
                                                  )
                distance_back = -(self.zero_mode_distance + self.image_location_offset - distance)
                H_back = get_propagation_kernel(
                                                nu = self.resolution[0] * 2,
                                                nv = self.resolution[1] * 2,
                                                dx = self.pixel_pitch,
                                                wavelength = self.wavelengths[channel_id],
                                                distance = distance_back,
                                                device = self.device,
                                                propagation_type = self.propagation_type,
                                                samples = self.aperture_samples,
                                                scale = self.resolution_factor
                                               )
                H = H_forward * H_back
            self.kernels[depth_id, channel_id] = H
            self.generated_kernels[depth_id, channel_id] = True
        else:
            H = self.kernels[depth_id, channel_id].detach().clone()
        field_scale = input_field
        field_scale_padded = zero_pad(field_scale)
        output_field_padded = custom(field_scale_padded, H, aperture = self.aperture)
        output_field = crop_center(output_field_padded)
        return output_field


    def reconstruct(self, hologram_phases, amplitude = None, no_grad = True, get_complex = False):
        """
        Internal function to reconstruct a given hologram.


        Parameters
        ----------
        hologram_phases            : torch.tensor
                                     Hologram phases [ch x m x n].
        amplitude                  : torch.tensor
                                     Amplitude profiles for each color primary [ch x m x n]
        no_grad                    : bool
                                     If set True, uses torch.no_grad in reconstruction.
        get_complex                : bool
                                     If set True, reconstructor returns the complex field but not the intensities.

        Returns
        -------
        reconstructions            : torch.tensor
                                     Reconstructed frames.
        """
        if no_grad:
            torch.no_grad()
        if len(hologram_phases.shape) > 3:
            hologram_phases = hologram_phases.squeeze(0)
        if get_complex == True:
            reconstruction_type = torch.complex64
        else:
            reconstruction_type = torch.float32
        reconstructions = torch.zeros(
                                      self.number_of_frames,
                                      self.number_of_depth_layers,
                                      self.number_of_channels,
                                      self.resolution[0] * self.resolution_factor,
                                      self.resolution[1] * self.resolution_factor,
                                      dtype = reconstruction_type,
                                      device = self.device
                                     )
        if isinstance(amplitude, type(None)):
            amplitude = torch.zeros(
                                    self.number_of_channels,
                                    self.resolution[0] * self.resolution_factor,
                                    self.resolution[1] * self.resolution_factor,
                                    device = self.device
                                   )
            amplitude[:, ::self.resolution_factor, ::self.resolution_factor] = 1.
        if self.resolution_factor != 1:
            hologram_phases_scaled = torch.zeros_like(amplitude)
            hologram_phases_scaled[
                                   :,
                                   ::self.resolution_factor,
                                   ::self.resolution_factor
                                  ] = hologram_phases
        else:
            hologram_phases_scaled = hologram_phases
        for frame_id in range(self.number_of_frames):
            for depth_id in range(self.number_of_depth_layers):
                for channel_id in range(self.number_of_channels):
                    laser_power = self.get_laser_powers()[frame_id][channel_id]
                    phase = hologram_phases_scaled[frame_id]
                    hologram = generate_complex_field(
                                                      laser_power * amplitude[channel_id],
                                                      phase * self.phase_scale[channel_id]
                                                     )
                    reconstruction_field = self.__call__(hologram, channel_id, depth_id)
                    if get_complex == True:
                        result = reconstruction_field
                    else:
                        result = calculate_amplitude(reconstruction_field) ** 2
                    reconstructions[
                                    frame_id,
                                    depth_id,
                                    channel_id
                                   ] = result.detach().clone()
        return reconstructions
