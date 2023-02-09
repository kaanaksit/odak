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
                 optimization_mode='SGD',
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
        self.optimization_mode = optimization_mode
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
        if self.optimization_mode == 'Stochastic Gradient Descent':
            self.optimizer = torch.optim.Adam([{'params': [self.phase, self.offset]}], lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.Adam([{'params': [self.phase]}], lr=self.learning_rate)


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


    def propagate(self, input_field, distance, padding=[True, True]):
        """
        Parameters
        ----------
        input_field         : torch.tensor
                              Input complex input field.
        distance            : float
                              Propagation distance.

        Returns
        -------
        output_field        : torch.tensor
                              Propagated output complex field.
        """
        if padding[0] == True:
            input_field_padded = zero_pad(input_field)
        elif padding[0] == False:
            input_field_padded = input_field
        output_field_padded = propagate_beam(
                                             input_field_padded,
                                             self.wavenumber,
                                             distance,
                                             self.slm_pixel_pitch,
                                             self.wavelength,
                                             self.propagation_type,
                                            )
        if padding[1] == True:
            output_field = crop_center(output_field_padded)
        elif padding[1] == False:
            output_field = output_field_padded
        return output_field


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
        if self.image_location == 0:
            location = self.zero_mode_distance
            distances = [
                         location,
                         -(location + residual)
                        ]
        else:
            location = self.image_location - residual
            distances = [0., location]
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
        if self.optimization_mode == 'Stochastic Gradient Descent':
            hologram = self.stochastic_gradient_descent()
        elif self.optimization_mode == 'Gerchberg-Saxton':
            hologram = self.gerchberg_saxton()
        elif self.optimization_mode == 'Double Phase':
            hologram = self.double_phase()
        else:
            hologram = self.double_phase()
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
        field = self.propagate(hologram, distances[0], padding=[pad[0], pad[1]])
        reconstruction = self.propagate(field, distances[1], padding=[pad[2], pad[3]])
        return reconstruction


    def gerchberg_saxton(self):
        """
        Function to optimize multiplane phase-only holograms using Gerchberg-Saxton algorithm.

        Returns
        -------
        hologram                   : torch.tensor
                                     Optimised hologram.

        """
        t = tqdm(range(self.number_of_iterations),leave=False)
        phase = torch.rand_like(self.phase)
        for step in t:
            total_loss = 0
            for plane_id in range(self.number_of_planes):
                torch.no_grad()
                distances = self.set_distances(plane_id)
                if self.image_location == 0:
                    distance = -distances[1]
                else:
                    distance = distances[1]
                hologram = generate_complex_field(self.amplitude, phase)
                reconstruction = self.propagate(hologram, distance, padding=[True, False])
                reconstruction_intensity = crop_center(calculate_amplitude(reconstruction)**2)
                new_target = (self.targets[plane_id].detach().clone() * self.mask)
                new_phase = calculate_phase(reconstruction)
                new_amplitude = calculate_amplitude(reconstruction)
                new_amplitude[
                              int(new_amplitude.shape[0]/2-new_target.shape[0]/2):int(new_amplitude.shape[0]/2+new_target.shape[0]/2),
                              int(new_amplitude.shape[1]/2-new_target.shape[1]/2):int(new_amplitude.shape[1]/2+new_target.shape[1]/2)
                             ] = new_target**0.5
                new_amplitude = torch.nan_to_num(new_amplitude, nan=0.0)
                reconstruction = generate_complex_field(new_amplitude, new_phase)
                hologram = self.propagate(reconstruction, -distance, padding=[False, True])
                phase = calculate_phase(hologram)
                loss = self.evaluate(
                                     reconstruction_intensity * self.mask,
                                     new_target * self.mask,
                                     plane_id
                                    ) 
                total_loss += loss
            description = "Gerchberg-Saxton, loss:{:.4f}".format(total_loss.item())
            t.set_description(description)
        print(description)
        if self.image_location == 0:
            start_location = distances[0]
            phase = shift_w_double_phase(
                                         phase,
                                         start_location,
                                         self.slm_pixel_pitch,
                                         self.wavelength,
                                         self.propagation_type
                                        )
        hologram = generate_complex_field(self.amplitude, phase)
        return hologram


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
                if self.image_location == 0: 
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
                else:
                   phase = self.phase
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


    def double_phase(self):
        """
        Generating a hologram with double phase coding. For more see: Maimone, Andrew, Andreas Georgiou, and Joel S. Kollin. "Holographic near-eye displays for virtual and augmented reality." ACM Transactions on Graphics (Tog) 36.4 (2017): 1-16.

        Returns
        -------
        hologram        : torch.tensor
                          Complex hologram.
        """
        total_field = torch.zeros(
                                  (self.phase.shape[0], self.phase.shape[1]), 
                                  dtype=torch.complex64,
                                  requires_grad=False
                                 ).to(self.device)
        phase = torch.rand_like(self.phase)
        for plane_id in range(self.number_of_planes):
            torch.no_grad()
            amplitude = self.targets[0].detach().clone()**0.5
            amplitude = torch.nan_to_num(amplitude, nan=0.0)
            field = generate_complex_field(
                                           amplitude,
                                           phase
                                          )
            phase = calculate_phase(self.model(field, [-self.image_spacing, 0]))
            distances = self.set_distances(plane_id)
            if self.image_location == 0:
                distance = distances[1]
            else:
                distance = -distances[1]
            at_hologram = self.model(field, [distance, 0])
            total_field = total_field + at_hologram.detach().clone() / torch.max(torch.abs(at_hologram))
        start_location = distances[0]
        phase = shift_w_double_phase(
                                     calculate_phase(total_field),
                                     start_location,
                                     self.slm_pixel_pitch,
                                     self.wavelength,
                                     self.propagation_type,
                                     amplitude=calculate_amplitude(total_field)
                                    )
        hologram = generate_complex_field(self.amplitude, phase)
        return hologram.detach().clone()


    def to(self, device):
        """
        Utilization function for setting the device.

        Parameters
        ----------
        device       : torch.device
                       Device to be used (e.g., CPU, Cuda, OpenCL).
        """
        self.device = device
        return self
