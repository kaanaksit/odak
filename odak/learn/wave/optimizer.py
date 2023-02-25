import torch
from tqdm import tqdm
from .util import wavenumber, generate_complex_field, calculate_amplitude, calculate_phase
from .classical import propagate_beam, shift_w_double_phase
from ..tools import zero_pad, crop_center


class multiplane_hologram_optimizer():
    """
    A highly configurable class for optimizing multiplane holograms.
    """
    def __init__(self, wavelength, image_location, 
                 image_spacing, slm_pixel_pitch,
                 slm_resolution, targets,
                 propagation_type='TR Fresnel', 
                 number_of_iterations=10, learning_rate=0.1,
                 phase_initial=None, amplitude_initial=None,
                 loss_function=None,
                 mask_limits=[0.2, 0.8, 0.05, 0.95],
                 number_of_planes=4,
                 zero_mode_distance=0.15,
                 device=None
                ):
        self.device = device
        if isinstance(self.device, type(None)):
            self.device = torch.device("cpu")
        torch.cuda.empty_cache()
        torch.random.seed()
        self.wavelength = wavelength
        self.image_location = image_location
        self.image_spacing = image_spacing
        self.slm_resolution = slm_resolution
        self.targets = targets
        self.slm_pixel_pitch = slm_pixel_pitch
        self.propagation_type = propagation_type
        self.mask_limits = mask_limits
        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate 
        self.number_of_planes = number_of_planes
        self.scene_center = self.image_spacing * (self.number_of_planes - 1) / 2.
        self.wavenumber = wavenumber(self.wavelength)
        self.zero_mode_distance = zero_mode_distance
        self.init_phase(phase_initial)
        self.init_amplitude(amplitude_initial)
        self.init_optimizer()
        self.init_mask()
        self.init_loss_function(loss_function)


    def init_amplitude(self,amplitude_initial):
        """
        Internal function to set the amplitude of the illumination source.
        """
        self.amplitude = amplitude_initial
        if isinstance(self.amplitude,type(None)):
            self.amplitude = torch.ones(
                                        self.slm_resolution[0],
                                        self.slm_resolution[1],
                                        requires_grad=False
                                       ).to(self.device)
        

    def init_phase(self,phase_initial):
        """
        Internal function to set the starting phase of the phase-only hologram.
        """
        self.phase = phase_initial
        if isinstance(self.phase, type(None)):
            self.phase = torch.rand(
                                    self.slm_resolution[0],
                                    self.slm_resolution[1]
                                   ).detach().to(self.device).requires_grad_()
            self.offset = torch.rand_like(self.phase)


    def init_optimizer(self):
        """
        Internal function to set the optimizer.
        """
        self.optimizer = torch.optim.Adam([{'params': [self.phase, self.offset]}], lr=self.learning_rate)


    def init_loss_function(self, loss_function=None, reduction='mean'):
        """
        Internal function to set the loss function.
        """
        self.loss_function = loss_function
        self.loss_type = 'other'
        if isinstance(self.loss_function, type(None)):
            self.loss_function = torch.nn.MSELoss(reduction=reduction)
            self.loss_type = 'naive'


    def init_mask(self):
        """
        Internal function to initialise the mask used in calculating the loss.
        """
        self.mask = torch.zeros(
                                self.slm_resolution[0],
                                self.slm_resolution[1],
                                requires_grad=False
                               ).to(self.device)
        self.mask[
                  int(self.slm_resolution[0]*self.mask_limits[0]):int(self.slm_resolution[0]*self.mask_limits[1]),
                  int(self.slm_resolution[1]*self.mask_limits[2]):int(self.slm_resolution[1]*self.mask_limits[3])
                 ] = 1


    def evaluate(self, input_image, target_image, plane_id):
        """
        Internal function to evaluate the loss.
        """
        if self.loss_type == 'naive':
            return self.loss_function(input_image, target_image)
        else:
            return self.loss_function(input_image, target_image, plane_id)


    def set_distances(self, plane_id, delta=0.0):
        """
        Internal function to set distances.

        Parameters
        ----------
        plane_id                    : int
                                      Plane number.
        delta                       : float
                                      A parameter to shift the image location.

        Returns
        -------
        distances                   : list
                                      List of distances.
        """
        residual = self.scene_center - plane_id * self.image_spacing + delta
        location = self.zero_mode_distance
        distances = [
                     location,
                     -(location + residual + self.image_location)
                    ]
        return distances


    def optimize(self):
        """
        Function to optimize multiplane phase-only holograms.

        Returns
        -------
        hologram_phase             : torch.tensor
                                     Phase of the optimized phase-only hologram.
        reconstruction_intensities : torch.tensor
                                     Intensities of the images reconstructed at each plane with the optimized phase-only hologram.
        """
        hologram = self.stochastic_gradient_descent()
        hologram_phase = calculate_phase(hologram)
        reconstruction_intensities = self.reconstruct(hologram_phase)
        return hologram_phase.detach().clone(), reconstruction_intensities.detach().clone()


    def reconstruct(self, hologram_phase):
        """
        Internal function to reconstruct a given hologram.


        Parameters
        ----------
        hologram_phase             : torch.tensor
                                     A monochrome hologram phase [mxn].
        
        Returns
        -------
        reconstruction_intensities : torch.tensor
                                     Reconstructed images.
        """
        hologram = generate_complex_field(self.amplitude, hologram_phase)
        torch.no_grad()
        reconstruction_intensities = torch.zeros(
                                                 self.number_of_planes,
                                                 self.phase.shape[0],
                                                 self.phase.shape[1],
                                                 requires_grad=False
                                                ).to(self.device)
        for plane_id in range(self.number_of_planes):
            distances = self.set_distances(plane_id)
            reconstruction = self.model(
                                        hologram,
                                        distances
                                       )
            reconstruction_intensities[plane_id] = calculate_amplitude(reconstruction)**2
        return reconstruction_intensities

 
    def model(self, hologram, distances, pad=[True, False, False, True]):
        """
        Internal function for forward and inverse models.
        """
        field = propagate_beam(
                               hologram,
                               self.wavenumber,
                               distances[0],
                               self.slm_pixel_pitch,
                               self.wavelength,
                               self.propagation_type,
                               zero_padding = [pad[0], False, pad[1]]
                              )
        reconstruction = propagate_beam(
                                        field,
                                        self.wavenumber,
                                        distances[1],
                                        self.slm_pixel_pitch,
                                        self.wavelength,
                                        self.propagation_type,
                                        zero_padding = [pad[2], False, pad[3]]
                                       )
        return reconstruction


    def stochastic_gradient_descent(self, delta=0.0):
        """
        Function to optimize multiplane phase-only holograms using stochastic gradient descent.

        Parameters
        ----------
        delta                      : float
                                     Incase you want to change the focus of the first plane, use this value in meters to electronically move the reconstructed image.

        Returns
        -------
        hologram                   : torch.tensor
                                     Optimised hologram.
        """
        t = tqdm(range(self.number_of_iterations),leave=False)
        for step in t:
            for plane_id in range(self.number_of_planes):
                self.optimizer.zero_grad()
                shifted_phase = self.phase
                phase_zero_mean = shifted_phase - torch.mean(shifted_phase)
                phase_offset = self.offset
                phase_low = phase_zero_mean - phase_offset
                phase_high = phase_zero_mean + phase_offset
                phase = torch.zeros_like(self.phase)
                phase[0::2, 0::2] = phase_low[0::2, 0::2]
                phase[0::2, 1::2] = phase_high[0::2, 1::2]
                phase[1::2, 0::2] = phase_high[1::2, 0::2]
                phase[1::2, 1::2] = phase_low[1::2, 1::2]
                hologram = generate_complex_field(self.amplitude, phase)
                distances = self.set_distances(plane_id)
                reconstruction = self.model(hologram, distances)
                reconstruction_intensity = calculate_amplitude(reconstruction)**2
                loss = self.evaluate(
                                     reconstruction_intensity * self.mask,
                                     self.targets[plane_id] * self.mask,
                                     plane_id
                                    )
                loss.backward(retain_graph=True)
                self.optimizer.step()
            description = "Stochastic Gradient Descent, loss:{:.4f}".format(loss.item())
            t.set_description(description)
        print(description)
        return hologram.detach().clone()

