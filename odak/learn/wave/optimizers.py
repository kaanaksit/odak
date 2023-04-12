import torch
from tqdm import tqdm
from .util import wavenumber, generate_complex_field, calculate_amplitude, calculate_phase
from .propagators import back_and_forth_propagator


class multiplane_hologram_optimizer():
    """
    A highly configurable class for optimizing multiplane holograms.
    """

    
    def __init__(self, wavelength, image_location, 
                 image_spacing, slm_pixel_pitch,
                 slm_resolution, targets,
                 propagation_type = 'TR Fresnel', 
                 number_of_iterations = 10, learning_rate = 0.1,
                 phase_initial = None, amplitude_initial = None,
                 loss_function = None,
                 mask_limits = [0.2, 0.8, 0.05, 0.95],
                 number_of_planes = 4,
                 zero_mode_distance = 0.15,
                 optimize_amplitude = False,
                 device = torch.device('cpu')
                ):
        self.device = device
        torch.cuda.empty_cache()
        torch.random.seed()
        self.wavelength = wavelength
        self.image_location = image_location
        self.image_spacing = image_spacing
        self.slm_resolution = slm_resolution
        self.targets = targets
        self.slm_pixel_pitch = slm_pixel_pitch
        self.model = back_and_forth_propagator(
                                               wavelength = self.wavelength,
                                               pixel_pitch = self.slm_pixel_pitch,
                                               device = self.device
                                              )
        self.propagation_type = propagation_type
        self.mask_limits = mask_limits
        self.number_of_iterations = number_of_iterations
        self.learning_rate = learning_rate 
        self.number_of_planes = number_of_planes
        self.scene_center = self.image_spacing * (self.number_of_planes - 1) / 2.
        self.wavenumber = wavenumber(self.wavelength)
        self.zero_mode_distance = zero_mode_distance
        self.optimize_amplitude = optimize_amplitude
        self.init_phase(phase_initial)
        self.init_amplitude(amplitude_initial)
        self.init_optimizer()
        self.init_mask()
        self.init_loss_function(loss_function)


    def init_amplitude(self, amplitude_initial):
        """
        Internal function to set the amplitude of the illumination source.
        """
        self.amplitude = amplitude_initial
        if isinstance(self.amplitude, type(None)):
            self.amplitude = torch.ones(
                                        self.slm_resolution[0],
                                        self.slm_resolution[1],
                                        requires_grad = False
                                       ).to(self.device)
        if self.optimize_amplitude == True:
            self.amplitude = self.amplitude.requires_grad_()
        

    def init_phase(self, phase_initial):
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
        parameters = [self.phase, self.offset]
        if self.optimize_amplitude == True:
            parameters.append(self.amplitude)
        self.optimizer = torch.optim.AdamW(parameters, lr = self.learning_rate)


    def init_loss_function(self, loss_function=None, reduction='mean'):
        """
        Internal function to set the loss function.
        """
        self.loss_function = loss_function
        self.loss_type = 'other'
        if isinstance(self.loss_function, type(None)):
            self.loss_function = torch.nn.MSELoss(reduction = reduction)
            self.loss_type = 'naive'


    def init_mask(self):
        """
        Internal function to initialise the mask used in calculating the loss.
        """
        self.mask = torch.zeros(
                                self.slm_resolution[0],
                                self.slm_resolution[1],
                                requires_grad = False,
                                device = self.device
                               )
        self.mask[
                  int(self.slm_resolution[0] * self.mask_limits[0]):int(self.slm_resolution[0] * self.mask_limits[1]),
                  int(self.slm_resolution[1] * self.mask_limits[2]):int(self.slm_resolution[1] * self.mask_limits[3])
                 ] = 1


    def evaluate(self, input_image, target_image, plane_id):
        """
        Internal function to evaluate the loss.
        """
        if self.loss_type == 'naive':
            return self.loss_function(input_image, target_image)
        else:
            return self.loss_function(input_image.unsqueeze(0), target_image, plane_id)


    def set_distances(self, plane_id):
        """
        Internal function to set distances.
        
        Parameters
        ----------
        plane_id                    : int
                                      Plane number.
        Returns
        -------
        distances                   : list
                                      List of distances.
        """
        residual = self.scene_center - plane_id * self.image_spacing
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
                                     Phase of the optimized hologram.
        hologram_amplitude         : torch.tensor
                                     Amplitude of the optimized hologram. 
        reconstruction_intensities : torch.tensor
                                     Intensities of the images reconstructed at each plane with the optimized phase-only hologram.
        """
        hologram = self.gradient_descent()
        hologram_phase = calculate_phase(hologram)
        hologram_amplitude = calculate_amplitude(hologram)
        reconstruction_intensities = self.reconstruct(hologram_amplitude, hologram_phase)
        return hologram_phase.detach().clone(), hologram_amplitude.detach().clone(), reconstruction_intensities.detach().clone()


    def reconstruct(self, hologram_amplitude, hologram_phase):
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
        hologram = generate_complex_field(hologram_amplitude, hologram_phase)
        torch.no_grad()
        reconstruction_intensities = torch.zeros(
                                                 self.number_of_planes,
                                                 self.phase.shape[0],
                                                 self.phase.shape[1],
                                                 requires_grad = False
                                                ).to(self.device)
        for plane_id in range(self.number_of_planes):
            distances = self.set_distances(plane_id)
            reconstruction = self.model(hologram, distances)
            reconstruction_intensities[plane_id] = calculate_amplitude(reconstruction) ** 2
        return reconstruction_intensities


    def amplitude_constrain(self, amplitude):
        """
        Function to limit amplitude of hologram between 0 and 1.
        
        Parameters 
        ----------
        amplitude                  : torch.tensor
                                     Input amplitude [m x n].
        
        Returns
        -------
        constrained_amplitude      : torch.tensor
                                     Constrained amplitude [m x n].
        """
        sigmoid = torch.nn.Sigmoid()
        constrained_amplitude = sigmoid(amplitude)
        return constrained_amplitude

    
    def double_phase_constrain(self, shifted_phase, phase_offset):
        """
        Function for generating double phase encoding alike phase-only holograms.
        
        Parameters
        ----------
        shifted_phase              : torch.tensor
                                     Input phase [m x n].
        phase_offset               : torch.tensor
                                     Input offset [m x n].
       
        Returns
        -------
        phase                      : torch.tensor
                                     Coded phase [m x n].
        """
        phase_zero_mean = shifted_phase - torch.mean(shifted_phase)
        phase_low = phase_zero_mean - phase_offset
        phase_high = phase_zero_mean + phase_offset
        phase = torch.zeros_like(shifted_phase)
        phase[0::2, 0::2] = phase_low[0::2, 0::2]
        phase[0::2, 1::2] = phase_high[0::2, 1::2]
        phase[1::2, 0::2] = phase_high[1::2, 0::2]
        phase[1::2, 1::2] = phase_low[1::2, 1::2]
        return phase


    def gradient_descent(self):
        """
        Function to optimize multiplane phase-only holograms using gradient descent.
        
        Returns
        -------
        hologram                   : torch.tensor
                                     Optimised hologram.
        """
        t = tqdm(range(self.number_of_iterations), leave = False, dynamic_ncols = True)
        for step in t:
            for plane_id in range(self.number_of_planes):
                self.optimizer.zero_grad()
                phase = self.double_phase_constrain(self.phase, self.offset)
                if self.optimize_amplitude == True:
                    amplitude = self.amplitude_constrain(self.amplitude)
                else:
                    amplitude = self.amplitude
                hologram = generate_complex_field(amplitude, phase)
                distances = self.set_distances(plane_id)
                reconstruction = self.model(hologram, distances)
                reconstruction_intensity = calculate_amplitude(reconstruction) ** 2
                loss = self.evaluate(
                                     reconstruction_intensity * self.mask,
                                     self.targets[plane_id] * self.mask,
                                     plane_id
                                    )
                loss.backward(retain_graph=True)
                self.optimizer.step()
            description = "Gradient Descent, loss:{:.4f}".format(loss.item())
            t.set_description(description)
        print(description)
        return hologram.detach().clone()
