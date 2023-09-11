import torch
import numpy as np
from ..tools import circular_binary_mask, zero_pad, crop_center
from .util import generate_complex_field, calculate_amplitude
from .classical import get_propagation_kernel


class holographic_display():
    """
    A class for simulating a holographic display.
    """
    def __init__(self, 
                 wavelengths, 
                 pixel_pitch = 3.74e-6,
                 resolution = [1920, 1080], 
                 volume_depth = 0.01,
                 number_of_depth_layers = 10,
                 image_location_offset = 0.005,
                 pinhole_size = 1500,
                 pad = [True, True],
                 illumination = None,
                 propagation_type = 'Bandlimited Angular Spectrum',
                 device = None
                ):
        """
        Parameters
        ----------
        wavelengths            : list
                                 List of wavelengths in meters (e.g., 531e-9).
        pixel_pitch            : float
                                 Pixel pitch in meters (e.g., 8e-6).
        resolution             : list
                                 Resolution (e.g., 1920 x 1080).
        volume_depth           : float
                                 Volume depth in meters.
        number_of_depth_layers : int
                                 Number of depth layers.
        image_location_offset  : float
                                 Image location offset in depth.
        pinhole_size           : int
                                 Size of the pinhole aperture in pixel in a 4f imaging system.
        pad                    : list
                                 Set it to list of True bools for zeropadding and cropping each time propagating (avoiding aliasing).
        illumination           : torch.tensor
                                 Provide the amplitude profile of the illumination source.
        device                 : torch.device
                                 Device to be used (e.g., cuda, cpu).
        """
        self.device = device
        if isinstance(self.device, type(None)):
            self.device = torch.device("cpu")
        self.pad = pad
        self.wavelengths = wavelengths
        self.resolution = resolution
        self.pixel_pitch = pixel_pitch
        self.volume_depth = volume_depth
        self.image_location_offset = torch.tensor(image_location_offset, device = device)
        self.number_of_depth_layers = number_of_depth_layers
        self.number_of_wavelengths = len(self.wavelengths)
        self.propagation_type = propagation_type
        self.pinhole_size = pinhole_size
        self.init_distances()
        self.init_amplitude(illumination)
        self.init_aperture()
        self.generate_kernels()

        
    def init_aperture(self):
        """
        Internal function to initialize aperture.
        """
        self.aperture = circular_binary_mask(
                                             self.resolution[0] * 2,
                                             self.resolution[1] * 2,
                                             self.pinhole_size,
                                            ).to(self.device) * 1.


    def init_amplitude(self, illumination):
        """
        Internal function to set the amplitude of the illumination source.
        """
        self.amplitude = torch.ones(
                                    self.resolution[0],
                                    self.resolution[1],
                                    requires_grad = False,
                                    device = self.device
                                   )
        if not isinstance(illumination, type(None)):
            self.amplitude = illumination


    def init_distances(self):
        """
        Internal function to set the image plane distances.
        """
        if self.number_of_depth_layers > 1:
            self.distances = torch.linspace(
                                            -self.volume_depth / 2., 
                                            self.volume_depth / 2., 
                                            self.number_of_depth_layers,
                                            device = self.device
                                           ) + self.image_location_offset
        else:
            self.distances = torch.tensor([self.image_location_offset], device = self.device)
       

    def forward(self, input_field, wavelength_id, depth_id):
        """

        Function that represents the forward model in hologram optimization.

        Parameters
        ----------
        input_field         : torch.tensor
                              Input complex input field.
        wavelength_id       : int
                              Identifying the color primary to be used.
        depth_id            : int
                              Identifying the depth layer to be used.

        Returns
        -------
        output_field        : torch.tensor
                              Propagated output complex field.
        """
        if self.pad[0]:
            input_field_padded = zero_pad(input_field)
        else:
            input_field_padded = input_field
        H = self.kernels[depth_id, wavelength_id].detach().clone()
        U_I = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(input_field_padded)))
        U_O = (U_I * self.aperture) * H
        output_field_padded = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(U_O)))
        if self.pad[1]:
            output_field = crop_center(output_field_padded)
        else:
            output_field = output_field_padded
        return output_field


    def generate_kernels(self):
        """
        Internal function to generate light transport kernels.
        """
        if self.pad[0]:
            multiplier = 2
        else:
            multiplier = 1         
        self.kernels = torch.zeros(
                                   self.number_of_depth_layers,
                                   self.number_of_wavelengths,
                                   self.resolution[0] * multiplier,
                                   self.resolution[1] * multiplier,
                                   device = self.device,
                                   dtype = torch.complex64
                                  )
        for distance_id, distance in enumerate(self.distances):
            for wavelength_id, wavelength in enumerate(self.wavelengths):
                 self.kernels[distance_id, wavelength_id] = get_propagation_kernel(
                                                                                   nu = self.kernels.shape[-2],
                                                                                   nv = self.kernels.shape[-1],
                                                                                   dx = self.pixel_pitch, 
                                                                                   wavelength = wavelength, 
                                                                                   distance = distance,
                                                                                   device = self.device,
                                                                                   propagation_type = self.propagation_type
                                                                                  )


    def reconstruct(self, hologram_phases, laser_powers):
        """
        Internal function to reconstruct a given hologram.


        Parameters
        ----------
        hologram_phases            : torch.tensor
                                     A monochrome hologram phase [m x n].
        laser_powers               : torch.tensor
                                     Laser powers for each hologram phase.
                                     Values must be between zero and one.
        
        Returns
        -------
        reconstruction_intensities : torch.tensor
                                     Reconstructed frames [w x k x l x m x n].
                                     First dimension represents the number of frames.
                                     Second dimension represents the depth layers.
                                     Third dimension is for the color primaries (each wavelength provided).
        """
        self.number_of_frames = hologram_phases.shape[0]
        reconstruction_intensities = torch.zeros(
                                                 self.number_of_frames,
                                                 self.number_of_depth_layers,
                                                 self.number_of_wavelengths,
                                                 self.resolution[0],
                                                 self.resolution[1],
                                                 device = self.device
                                                )
        for frame_id in range(self.number_of_frames): 
            for depth_id in range(self.number_of_depth_layers): 
                for wavelength_id in range(self.number_of_wavelengths):
                    laser_power = laser_powers[frame_id][wavelength_id]
                    hologram = generate_complex_field(
                                                      laser_power * self.amplitude, 
                                                      hologram_phases[frame_id]
                                                     )
                    reconstruction_field = self.forward(hologram, wavelength_id, depth_id)
                    reconstruction_intensities[frame_id, depth_id, wavelength_id] = calculate_amplitude(reconstruction_field) ** 2
        return reconstruction_intensities
