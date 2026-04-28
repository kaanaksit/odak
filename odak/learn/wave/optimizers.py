import torch
from tqdm import tqdm
from .propagators import propagator
from .util import (
    wavenumber,
    generate_complex_field,
    calculate_amplitude,
    calculate_phase,
)
from ..tools import torch_load, multi_scale_total_variation_loss, quantize, circular_binary_mask, spatial_gradient
from ...log import logger


class multi_color_hologram_optimizer:
    """
    A class for optimizing single or multi color holograms.
    For more details, see Kavaklı et al., SIGGRAPH ASIA 2023, Multi-color Holograms Improve Brightness in HOlographic Displays.
    """

    def __init__(
        self,
        wavelengths,
        resolution,
        targets,
        propagator,
        number_of_frames=3,
        number_of_depth_layers=1,
        learning_rate=2e-2,
        scheduler_power=1,
        double_phase=True,
        scale_factor=1,
        method="multi-color",
        channel_power_filename="",
        device=None,
        loss_function=None,
        peak_amplitude=1.0,
        optimize_peak_amplitude=False,
        img_loss_thres=2e-3,
        reduction="sum",
    ):
        self.device = device
        if isinstance(self.device, type(None)):
            self.device = torch.device("cpu")
        torch.cuda.empty_cache()
        torch.random.seed()
        self.wavelengths = wavelengths
        self.resolution = resolution
        self.targets = targets
        if propagator.propagation_type != "Impulse Response Fresnel":
            scale_factor = 1
        self.scale_factor = scale_factor
        self.propagator = propagator
        self.learning_rate = learning_rate
        self.scheduler_power = scheduler_power
        self.number_of_channels = len(self.wavelengths)
        self.number_of_frames = number_of_frames
        self.number_of_depth_layers = number_of_depth_layers
        self.double_phase = double_phase
        self.channel_power_filename = channel_power_filename
        self.method = method
        if self.method != "conventional" and self.method != "multi-color":
            logger.warning(
                "Unknown optimization method. Options are conventional or multi-color."
            )
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
        self.init_loss_function(loss_function, reduction=reduction)
        self.init_amplitude()
        self.init_phase_scale()

    def init_peak_amplitude_scale(self):
        """
        Internal function to set the phase scale.
        """
        self.peak_amplitude = torch.tensor(
            self.peak_amplitude, requires_grad=True, device=self.device
        )

    def init_phase_scale(self):
        """
        Internal function to set the phase scale.
        """
        if self.method == "conventional":
            self.phase_scale = torch.tensor(
                [1.0, 1.0, 1.0], requires_grad=False, device=self.device
            )
        if self.method == "multi-color":
            self.phase_scale = torch.tensor(
                [1.0, 1.0, 1.0], requires_grad=False, device=self.device
            )

    def init_amplitude(self):
        """
        Internal function to set the amplitude of the illumination source.
        """
        self.amplitude = torch.zeros(
            self.resolution[0] * self.scale_factor,
            self.resolution[1] * self.scale_factor,
            requires_grad=False,
            device=self.device,
        )
        self.amplitude[:: self.scale_factor, :: self.scale_factor] = 1.0

    def init_phase(self):
        """
        Internal function to set the starting phase of the phase-only hologram.
        """
        self.phase = torch.randn(
            self.number_of_frames,
            self.resolution[0],
            self.resolution[1],
            device=self.device,
            requires_grad=False,
        ) * 2. * torch.pi
        self.phase.requires_grad = True
        self.offset = torch.randn(
            self.number_of_frames, requires_grad=False, device=self.device
        )

    def init_channel_power(self):
        """
        Internal function to set the starting phase of the phase-only hologram.
        """
        if self.method == "conventional":
            logger.warning("Scheme: Conventional")
            self.channel_power = torch.eye(
                self.number_of_frames,
                self.number_of_channels,
                device=self.device,
                requires_grad=False,
            )

        elif self.method == "multi-color":
            logger.warning("Scheme: Multi-color")
            self.channel_power = torch.ones(
                self.number_of_frames,
                self.number_of_channels,
                device=self.device,
                requires_grad=True,
            )
        if self.channel_power_filename != "":
            self.channel_power = torch_load(self.channel_power_filename).to(self.device)
            self.channel_power.requires_grad = False
            self.channel_power[self.channel_power < 0.0] = 0.0
            self.channel_power[self.channel_power > 1.0] = 1.0
            if self.method == "multi-color":
                self.channel_power.requires_grad = True
            if self.method == "conventional":
                self.channel_power = torch.abs(torch.cos(self.channel_power))
            logger.warning("Channel powers:")
            logger.warning(self.channel_power)
            logger.warning(
                "Channel powers loaded from {}.".format(self.channel_power_filename)
            )
        self.propagator.set_laser_powers(self.channel_power)

    def init_optimizer(self, number_of_iterations=100):
        """
        Internal function to set the optimizer.
        """
        optimization_variables = [self.phase]
        if self.optimize_peak_amplitude:
            optimization_variables.append(self.peak_amplitude)
        if self.method == "multi-color":
            optimization_variables.append(self.propagator.channel_power)
        self.optimizer = torch.optim.Adam(optimization_variables, lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
            self.optimizer,
            total_iters=number_of_iterations * 4,
            power=self.scheduler_power,
            last_epoch=-1,
        )

    def init_loss_function(self, loss_function, reduction="sum"):
        """
        Internal function to set the loss function.
        """
        self.l2_loss = torch.nn.MSELoss(reduction=reduction)
        self.loss_type = "custom"
        self.loss_function = loss_function
        if isinstance(self.loss_function, type(None)):
            self.loss_type = "conventional"
            self.loss_function = torch.nn.MSELoss(reduction=reduction)

    def evaluate(
        self,
        input_image,
        target_image,
        plane_id=0,
        noise_ratio=1e-3,
        inject_noise=False,
    ):
        """
        Internal function to evaluate the loss.
        """
        if self.loss_type == "conventional":
            loss = self.loss_function(input_image, target_image)
        elif self.loss_type == "custom":
            loss = 0
            for i in range(len(self.wavelengths)):
                loss += self.loss_function(
                    input_image[i],
                    target_image[i],
                    plane_id=plane_id,
                    noise_ratio=noise_ratio,
                    inject_noise=inject_noise,
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
        phase_low = torch.nan_to_num(phase_zero_mean - phase_offset, nan=2 * torch.pi)
        phase_high = torch.nan_to_num(phase_zero_mean + phase_offset, nan=2 * torch.pi)
        phase_only = torch.zeros_like(phase)
        phase_only[0::2, 0::2] = phase_low[0::2, 0::2]
        phase_only[0::2, 1::2] = phase_high[0::2, 1::2]
        phase_only[1::2, 0::2] = phase_high[1::2, 0::2]
        phase_only[1::2, 1::2] = phase_low[1::2, 1::2]
        return phase_only

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
        phase_only = torch.nan_to_num(phase - phase_offset, nan=2 * torch.pi)
        return phase_only

    def eyebox_constrain(self, phase, offset, diameter):
        """
        Internal function to calculate the mean of spatial gradients of the masked FFT amplitude.

        Parameters
        ----------
        phase                      : torch.tensor
                                     Input phase values (shape: [height, width]).
        offset                     : tuple or float
                                     Offset (x, y) in pixel values for the circular aperture center.
        diameter                   : float
                                     Diameter of the circular aperture in pixel values.

        Returns
        -------
        gradient_mean              : torch.tensor
                                     Mean of the spatial gradients of the masked FFT amplitude.
        """
        complex_field = generate_complex_field(torch.ones_like(phase), phase)
        fft_field = torch.fft.fftshift(torch.fft.fft2(complex_field))
        fft_amplitude = torch.abs(fft_field)
        height, width = fft_amplitude.shape
        radius = diameter / 2.0
        mask = circular_binary_mask(height, width, radius, offset_x=offset[0], offset_y=offset[1]).to(self.device).squeeze(0).squeeze(0)
        masked_amplitude = fft_amplitude * mask
        masked_amplitude = masked_amplitude / masked_amplitude.max()
        masked_amplitude_std = torch.std(masked_amplitude)
        masked_amplitude_peak = masked_amplitude.max() - masked_amplitude.mean()
        unmask = torch.abs(1.0 - mask)
        unmasked_amplitude = fft_amplitude * unmask
        unmask_mean = unmasked_amplitude.mean()
        gradient_mean = multi_scale_total_variation_loss(masked_amplitude, levels=3)
        masked_amplitude_mean = (1.0 - masked_amplitude.mean())
        loss = gradient_mean + masked_amplitude_mean + masked_amplitude_std + masked_amplitude_peak + unmask_mean * 1e-2
        return loss

    def gradient_descent(
        self,
        number_of_iterations=100,
        weights=[1.0, 1.0, 0.0],
        inject_noise=False,
        eyebox={
            "offset" : 0.0,
            "diameter" : 100
            },
        noise_ratio=1e-3,
    ):
        """
        Function to optimize multiplane phase-only holograms using stochastic gradient descent.

        Parameters
        ----------
        number_of_iterations       : float
                                     Number of iterations.
        weights                    : list
                                     Weights used in the loss function.
        inject_noise               : bool
                                     When set True, this will inject noise with the given `noise_ratio` to the target images.
        noise_ratio                : float
                                     Noise ratio, a multiplier (1e-3 is 0.1 percent).

        Returns
        -------
        hologram                   : torch.tensor
                                     Optimised hologram.
        """
        hologram_phases = torch.zeros(
            self.number_of_frames,
            self.resolution[0],
            self.resolution[1],
            device=self.device,
        )
        t = tqdm(range(number_of_iterations), leave=False, dynamic_ncols=True)
        if self.optimize_peak_amplitude:
            peak_amp_cache = self.peak_amplitude.item()
        for step in t:
            for g in self.optimizer.param_groups:
                learning_rate = g["lr"]
            total_loss = 0
            t_depth = tqdm(
                range(self.targets.shape[0]), leave=False, dynamic_ncols=True
            )
            for depth_id in t_depth:
                self.optimizer.zero_grad()
                depth_target = self.targets[depth_id]
                reconstruction_intensities = torch.zeros(
                    self.number_of_frames,
                    self.number_of_channels,
                    self.resolution[0] * self.scale_factor,
                    self.resolution[1] * self.scale_factor,
                    device=self.device,
                )
                loss_laser = 0.0    
                loss_eyebox = 0.0
                laser_powers = self.propagator.get_laser_powers()
                for frame_id in range(self.number_of_frames):
                    self.offset[frame_id] = self.phase[frame_id].detach().clone().mean()
                    if self.double_phase:
                        phase = self.double_phase_constrain(
                            self.phase[frame_id], self.offset[frame_id]
                        )
                    else:
                        phase = self.direct_phase_constrain(
                            self.phase[frame_id], self.offset[frame_id]
                        )
                    if weights[2] > 0.0:
                        loss_eyebox += self.eyebox_constrain(phase, offset=eyebox['offset'], diameter=eyebox['diameter'])
                    phase_wrapped = phase % (2. * torch.pi)
                    for channel_id in range(self.number_of_channels):
                        phase_scaled = torch.zeros_like(self.amplitude)
                        phase_scaled[:: self.scale_factor, :: self.scale_factor] = phase_wrapped
                        laser_power = laser_powers[frame_id][channel_id]
                        hologram = generate_complex_field(
                            laser_power * self.amplitude,
                            phase_scaled * self.phase_scale[channel_id],
                        )
                        reconstruction_field = self.propagator(
                            hologram, channel_id, depth_id
                        )
                        intensity = calculate_amplitude(reconstruction_field) ** 2
                        reconstruction_intensities[frame_id, channel_id] += intensity.squeeze(0).squeeze(0)
                    hologram_phases[frame_id] = phase_wrapped.detach().clone()
                if weights[1] > 0.0:
                    loss_laser += self.l2_loss(
                        torch.amax(depth_target, dim=(1, 2)) * self.peak_amplitude,
                        torch.sum(laser_powers, dim=0),
                    )
                    loss_laser += self.l2_loss(
                        torch.tensor([self.number_of_frames * self.peak_amplitude]).to(
                            self.device
                        ),
                        torch.sum(laser_powers).view(1,),
                    )
                    loss_laser += torch.cos(torch.min(torch.sum(laser_powers, dim=1)))
                reconstruction_intensity = torch.sum(reconstruction_intensities, dim=0)
                loss_image = self.evaluate(
                    reconstruction_intensity,
                    depth_target * self.peak_amplitude,
                    noise_ratio=noise_ratio,
                    inject_noise=inject_noise,
                    plane_id=depth_id,
                )
                loss = weights[0] * loss_image
                if weights[1] > 0.0:
                    loss += weights[1] * loss_laser
                if weights[2] > 0.0:
                    loss += weights[3] * loss_eyebox
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.detach().item()
                loss_image = loss_image.detach()
                del loss_laser
                del loss
            description = "Loss:{:.3f} Loss Image:{:.3f} Peak Amp:{:.1f} Learning rate:{:.4f}".format(
                total_loss, loss_image.item(), self.peak_amplitude, learning_rate
            )
            t.set_description(description)
            del total_loss
            del loss_image
            del reconstruction_field
            del reconstruction_intensities
            del intensity
            del phase
            del hologram
        logger.warning(description)
        return hologram_phases.detach()

    def optimize(
        self,
        number_of_iterations=100,
        weights=[1.0, 1.0, 1.0],
        eyebox={
            "offset" : [0.0, 0.0],
            "diameter" : 100
            },
        bits=8,
        inject_noise=False,
        noise_ratio=1e-3,
    ):
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
        inject_noise               : bool
                                     When set True, this will inject noise with the given `noise_ratio` to the target images.
        noise_ratio                : float
                                     Noise ratio, a multiplier (1e-3 is 0.1 percent).


        Returns
        -------
        hologram_phases            : torch.tensor
                                     Phases of the optimized phase-only hologram.
        reconstruction_intensities : torch.tensor
                                     Intensities of the images reconstructed at each plane with the optimized phase-only hologram.
        """
        self.init_optimizer(number_of_iterations=number_of_iterations)
        hologram_phases = self.gradient_descent(
            number_of_iterations=number_of_iterations,
            noise_ratio=noise_ratio,
            inject_noise=inject_noise,
            eyebox=eyebox,
            weights=weights,
        )
        hologram_phases = (
            quantize(
                hologram_phases % (2 * torch.pi), bits=bits, limits=[0.0, 2 * torch.pi]
            )
            / 2**bits
            * 2
            * torch.pi
        )
        torch.no_grad()
        reconstruction_intensities = self.propagator.reconstruct(hologram_phases)
        laser_powers = self.propagator.get_laser_powers()
        channel_powers = self.propagator.channel_power
        logger.warning("Final peak amplitude: {}".format(self.peak_amplitude))
        logger.warning("Laser powers: {}".format(laser_powers))
        return (
            hologram_phases,
            reconstruction_intensities,
            laser_powers,
            channel_powers,
            float(self.peak_amplitude),
        )
