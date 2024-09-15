import torch
import logging
import numpy as np
from tqdm import tqdm
from .util import wavenumber, generate_complex_field, calculate_amplitude, calculate_phase
from ..tools import torch_load, multi_scale_total_variation_loss, quantize
from .propagators import propagator


class multi_color_hologram_optimizer():
    """
    A class for optimizing single or multi color holograms.
    For more details, see Kavaklı et al., SIGGRAPH ASIA 2023, Multi-color Holograms Improve Brightness in HOlographic Displays.
    """
    def __init__(self,
                 wavelengths,
                 resolution,
                 targets,
                 propagator,
                 number_of_frames = 3,
                 number_of_depth_layers = 1,
                 learning_rate = 2e-2,
                 learning_rate_floor = 5e-3,
                 double_phase = True,
                 scale_factor = 1,
                 method = 'multi-color',
                 channel_power_filename = '',
                 device = None,
                 loss_function = None,
                 peak_amplitude = 1.0,
                 optimize_peak_amplitude = False,
                 img_loss_thres = 2e-3,
                 reduction = 'sum'
                ):
        self.device = device
        if isinstance(self.device, type(None)):
            self.device = torch.device("cpu")
        torch.cuda.empty_cache()
        torch.random.seed()
        self.wavelengths = wavelengths
        self.resolution = resolution
        self.targets = targets
        if propagator.propagation_type != 'Impulse Response Fresnel':
            scale_factor = 1
        self.scale_factor = scale_factor
        self.propagator = propagator
        self.learning_rate = learning_rate
        self.learning_rate_floor = learning_rate_floor
        self.number_of_channels = len(self.wavelengths)
        self.number_of_frames = number_of_frames
        self.number_of_depth_layers = number_of_depth_layers
        self.double_phase = double_phase
        self.channel_power_filename = channel_power_filename
        self.method = method
        if self.method != 'conventional' and self.method != 'multi-color':
           logging.warning('Unknown optimization method. Options are conventional or multi-color.')
           import sys
           sys.exit()
        self.peak_amplitude = peak_amplitude
        self.optimize_peak_amplitude = optimize_peak_amplitude
        if self.optimize_peak_amplitude:
            self.init_peak_amplitude_scale()
        self.img_loss_thres = img_loss_thres
        self.kernels = []
        self.init_phase()
        self.init_channel_power()
        self.init_loss_function(loss_function, reduction = reduction)
        self.init_amplitude()
        self.init_phase_scale()


    def init_peak_amplitude_scale(self):
        """
        Internal function to set the phase scale.
        """
        self.peak_amplitude = torch.tensor(
                                           self.peak_amplitude,
                                           requires_grad = True,
                                           device=self.device
                                          )


    def init_phase_scale(self):
        """
        Internal function to set the phase scale.
        """
        if self.method == 'conventional':
            self.phase_scale = torch.tensor(
                                            [
                                             1.,
                                             1.,
                                             1.
                                            ],
                                            requires_grad = False,
                                            device = self.device
                                           )
        if self.method == 'multi-color':
            self.phase_scale = torch.tensor(
                                            [
                                             1.,
                                             1.,
                                             1.
                                            ],
                                            requires_grad = False,
                                            device = self.device
                                           )


    def init_amplitude(self):
        """
        Internal function to set the amplitude of the illumination source.
        """
        self.amplitude = torch.zeros(
                                     self.resolution[0] * self.scale_factor,
                                     self.resolution[1] * self.scale_factor,
                                     requires_grad = False,
                                     device = self.device
                                    )
        self.amplitude[::self.scale_factor, ::self.scale_factor] = 1.


    def init_phase(self):
        """
        Internal function to set the starting phase of the phase-only hologram.
        """
        self.phase = torch.zeros(
                                 self.number_of_frames,
                                 self.resolution[0],
                                 self.resolution[1],
                                 device = self.device,
                                 requires_grad = True
                                )
        self.offset = torch.rand_like(self.phase, requires_grad = True, device = self.device)


    def init_channel_power(self):
        """
        Internal function to set the starting phase of the phase-only hologram.
        """
        if self.method == 'conventional':
            logging.warning('Scheme: Conventional')
            self.channel_power = torch.eye(
                                           self.number_of_frames,
                                           self.number_of_channels,
                                           device = self.device,
                                           requires_grad = False
                                          )

        elif self.method == 'multi-color':
            logging.warning('Scheme: Multi-color')
            self.channel_power = torch.ones(
                                            self.number_of_frames,
                                            self.number_of_channels,
                                            device = self.device,
                                            requires_grad = True
                                           )
        if self.channel_power_filename != '':
            self.channel_power = torch_load(self.channel_power_filename).to(self.device)
            self.channel_power.requires_grad = False
            self.channel_power[self.channel_power < 0.] = 0.
            self.channel_power[self.channel_power > 1.] = 1.
            if self.method == 'multi-color':
                self.channel_power.requires_grad = True
            if self.method == 'conventional':
                self.channel_power = torch.abs(torch.cos(self.channel_power))
            logging.warning('Channel powers:')
            logging.warning(self.channel_power)
            logging.warning('Channel powers loaded from {}.'.format(self.channel_power_filename))
        self.propagator.set_laser_powers(self.channel_power)



    def init_optimizer(self):
        """
        Internal function to set the optimizer.
        """
        optimization_variables = [self.phase, self.offset]
        if self.optimize_peak_amplitude:
            optimization_variables.append(self.peak_amplitude)
        if self.method == 'multi-color':
            optimization_variables.append(self.propagator.channel_power)
        self.optimizer = torch.optim.Adam(optimization_variables, lr=self.learning_rate)


    def init_loss_function(self, loss_function, reduction = 'sum'):
        """
        Internal function to set the loss function.
        """
        self.l2_loss = torch.nn.MSELoss(reduction = reduction)
        self.loss_type = 'custom'
        self.loss_function = loss_function
        if isinstance(self.loss_function, type(None)):
            self.loss_type = 'conventional'
            self.loss_function = torch.nn.MSELoss(reduction = reduction)



    def evaluate(self, input_image, target_image, plane_id = 0):
        """
        Internal function to evaluate the loss.
        """
        if self.loss_type == 'conventional':
            loss = self.loss_function(input_image, target_image)
        elif self.loss_type == 'custom':
            loss = 0
            for i in range(len(self.wavelengths)):
                loss += self.loss_function(
                                           input_image[i],
                                           target_image[i],
                                           plane_id = plane_id
                                          )
        return loss


    def double_phase_constrain(self, phase, phase_offset):
        """
        Internal function to constrain a given phase similarly to double phase encoding.

        Parameters
        ----------
        phase                      : torch.tensor
                                     Input phase values to be constrained.
        phase_offset               : torch.tensor
                                     Input phase offset value.

        Returns
        -------
        phase_only                 : torch.tensor
                                     Constrained output phase.
        """
        phase_zero_mean = phase - torch.mean(phase)
        phase_low = torch.nan_to_num(phase_zero_mean - phase_offset, nan = 2 * np.pi)
        phase_high = torch.nan_to_num(phase_zero_mean + phase_offset, nan = 2 * np.pi)
        loss = multi_scale_total_variation_loss(phase_low, levels = 6)
        loss += multi_scale_total_variation_loss(phase_high, levels = 6)
        loss += torch.std(phase_low)
        loss += torch.std(phase_high)
        phase_only = torch.zeros_like(phase)
        phase_only[0::2, 0::2] = phase_low[0::2, 0::2]
        phase_only[0::2, 1::2] = phase_high[0::2, 1::2]
        phase_only[1::2, 0::2] = phase_high[1::2, 0::2]
        phase_only[1::2, 1::2] = phase_low[1::2, 1::2]
        return phase_only, loss


    def direct_phase_constrain(self, phase, phase_offset):
        """
        Internal function to constrain a given phase.

        Parameters
        ----------
        phase                      : torch.tensor
                                     Input phase values to be constrained.
        phase_offset               : torch.tensor
                                     Input phase offset value.

        Returns
        -------
        phase_only                 : torch.tensor
                                     Constrained output phase.
        """
        phase_only = torch.nan_to_num(phase - phase_offset, nan = 2 * np.pi)
        loss = multi_scale_total_variation_loss(phase, levels = 6)
        loss += multi_scale_total_variation_loss(phase_offset, levels = 6)
        return phase_only, loss


    def gradient_descent(self, number_of_iterations=100, weights=[1., 1., 0., 0.]):
        """
        Function to optimize multiplane phase-only holograms using stochastic gradient descent.

        Parameters
        ----------
        number_of_iterations       : float
                                     Number of iterations.
        weights                    : list
                                     Weights used in the loss function.

        Returns
        -------
        hologram                   : torch.tensor
                                     Optimised hologram.
        """
        hologram_phases = torch.zeros(
                                      self.number_of_frames,
                                      self.resolution[0],
                                      self.resolution[1],
                                      device = self.device
                                     )
        t = tqdm(range(number_of_iterations), leave = False, dynamic_ncols = True)
        if self.optimize_peak_amplitude:
            peak_amp_cache = self.peak_amplitude.item()
        for step in t:
            for g in self.optimizer.param_groups:
                g['lr'] -= (self.learning_rate - self.learning_rate_floor) / number_of_iterations
                if g['lr'] < self.learning_rate_floor:
                    g['lr'] = self.learning_rate_floor
                learning_rate = g['lr']
            total_loss = 0
            t_depth = tqdm(range(self.targets.shape[0]), leave = False, dynamic_ncols = True)
            for depth_id in t_depth:
                self.optimizer.zero_grad()
                depth_target = self.targets[depth_id]
                reconstruction_intensities = torch.zeros(
                                                         self.number_of_frames,
                                                         self.number_of_channels,
                                                         self.resolution[0] * self.scale_factor,
                                                         self.resolution[1] * self.scale_factor,
                                                         device = self.device
                                                        )
                loss_variation_hologram = 0
                laser_powers = self.propagator.get_laser_powers()
                for frame_id in range(self.number_of_frames):
                    if self.double_phase:
                        phase, loss_phase = self.double_phase_constrain(
                                                                        self.phase[frame_id],
                                                                        self.offset[frame_id]
                                                                       )
                    else:
                        phase, loss_phase = self.direct_phase_constrain(
                                                                        self.phase[frame_id],
                                                                        self.offset[frame_id]
                                                                       )
                    loss_variation_hologram += loss_phase
                    for channel_id in range(self.number_of_channels):
                        phase_scaled = torch.zeros_like(self.amplitude)
                        phase_scaled[::self.scale_factor, ::self.scale_factor] = phase
                        laser_power = laser_powers[frame_id][channel_id]
                        hologram = generate_complex_field(
                                                          laser_power * self.amplitude,
                                                          phase_scaled * self.phase_scale[channel_id]
                                                         )
                        reconstruction_field = self.propagator(hologram, channel_id, depth_id)
                        intensity = calculate_amplitude(reconstruction_field) ** 2
                        reconstruction_intensities[frame_id, channel_id] += intensity
                    hologram_phases[frame_id] = phase.detach().clone()
                loss_laser = self.l2_loss(
                                          torch.amax(depth_target, dim = (1, 2)) * self.peak_amplitude,
                                          torch.sum(laser_powers, dim = 0)
                                         )
                loss_laser += self.l2_loss(
                                           torch.tensor([self.number_of_frames * self.peak_amplitude]).to(self.device),
                                           torch.sum(laser_powers).view(1,)
                                          )
                loss_laser += torch.cos(torch.min(torch.sum(laser_powers, dim = 1)))
                reconstruction_intensity = torch.sum(reconstruction_intensities, dim=0)
                loss_image = self.evaluate(
                                           reconstruction_intensity,
                                           depth_target * self.peak_amplitude,
                                           plane_id = depth_id
                                          )
                loss = weights[0] * loss_image
                loss += weights[1] * loss_laser
                loss += weights[2] * loss_variation_hologram
                include_pa_loss_flag = self.optimize_peak_amplitude and loss_image < self.img_loss_thres
                if include_pa_loss_flag:
                    loss -= self.peak_amplitude * 1.
                if self.method == 'conventional':
                    loss.backward()
                else:
                    loss.backward(retain_graph = True)
                self.optimizer.step()
                if include_pa_loss_flag:
                    peak_amp_cache = self.peak_amplitude.item()
                else:
                    with torch.no_grad():
                        if self.optimize_peak_amplitude:
                            self.peak_amplitude.view([1])[0] = peak_amp_cache
                total_loss += loss.detach().item()
                loss_image = loss_image.detach()
                del loss_laser
                del loss_variation_hologram
                del loss
            description = "Loss:{:.3f} Loss Image:{:.3f} Peak Amp:{:.1f} Learning rate:{:.4f}".format(total_loss, loss_image.item(), self.peak_amplitude, learning_rate)
            t.set_description(description)
            del total_loss
            del loss_image
            del reconstruction_field
            del reconstruction_intensities
            del intensity
            del phase
            del hologram
        logging.warning(description)
        return hologram_phases.detach()


    def optimize(self, number_of_iterations=100, weights=[1., 1., 1.], bits = 8):
        """
        Function to optimize multiplane phase-only holograms.

        Parameters
        ----------
        number_of_iterations       : int
                                     Number of iterations.
        weights                    : list
                                     Loss weights.
        bits                       : int
                                     Quantizes the hologram using the given bits and reconstructs.

        Returns
        -------
        hologram_phases            : torch.tensor
                                     Phases of the optimized phase-only hologram.
        reconstruction_intensities : torch.tensor
                                     Intensities of the images reconstructed at each plane with the optimized phase-only hologram.
        """
        self.init_optimizer()
        hologram_phases = self.gradient_descent(
                                                number_of_iterations=number_of_iterations,
                                                weights=weights
                                               )
        hologram_phases = quantize(hologram_phases % (2 * np.pi), bits = bits, limits = [0., 2 * np.pi]) / 2 ** bits * 2 * np.pi
        torch.no_grad()
        reconstruction_intensities = self.propagator.reconstruct(hologram_phases)
        laser_powers = self.propagator.get_laser_powers()
        channel_powers = self.propagator.channel_power
        logging.warning("Final peak amplitude: {}".format(self.peak_amplitude))
        logging.warning('Laser powers: {}'.format(laser_powers))
        return hologram_phases, reconstruction_intensities, laser_powers, channel_powers, float(self.peak_amplitude)
