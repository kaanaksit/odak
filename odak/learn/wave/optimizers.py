import torch
from tqdm import tqdm
from .propagators import propagator
from .util import (
    wavenumber,
    generate_complex_field,
    calculate_amplitude,
    calculate_phase,
    compose_double_phase,
    decompose_double_phase,
)
from ..tools import torch_load, multi_scale_total_variation_loss, quantize, circular_binary_mask, spatial_gradient
from ...log import logger


class multi_color_hologram_optimizer:
    """
    A class for optimizing single or multi color holograms.
    For more details, see Kavaklı et al., SIGGRAPH ASIA 2023, Multi-color Holograms Improve Brightness in Holographic Displays.

    Key methods:
    - optimize: Main entry point for optimization and quantization.
    - gradient_descent: Core SGD optimization loop.
    - evaluate: Loss calculation for reconstructed images.

    Attributes
    ------
    device                     : torch.device
                                Device to run optimization on.
    wavelengths                : list
                                List of wavelengths for optimization.
    resolution                 : tuple
                                Resolution (height, width) of the hologram.
    targets                    : torch.tensor
                                Target images for optimization.
    propagator                 : propagator
                                Wave propagation object.
    scale_factor               : int
                                Scaling factor for hologram resolution.
    learning_rate              : float
                                Learning rate for the optimizer.
    scheduler_power            : int
                                Power for polynomial learning rate scheduler.
    number_of_channels         : int
                                Number of wavelength channels.
    number_of_frames           : int
                                Number of temporal frames.
    number_of_depth_layers     : int
                                Number of depth layers.
    channel_power_filename     : str
                                Filename to load channel powers from.
    method                     : str
                                Optimization method ("conventional" or "multi-color").
    peak_amplitude             : float or torch.tensor
                                Peak amplitude for reconstruction (learnable if optimize_peak_amplitude=True).
    optimize_peak_amplitude    : bool
                                Whether to optimize peak amplitude.
    img_loss_thres             : float
                                Image loss threshold.
    kernels                    : list
                                List of kernels (unused).
    phase                      : torch.tensor
                                Hologram phase tensor (requires_grad=True).
    offset                     : torch.tensor
                                Phase offset for each frame (requires_grad=True).
    channel_power              : torch.tensor
                                Channel power distribution.
    phase_scale                : torch.tensor
                                Phase scale for each wavelength channel.
    amplitude                  : torch.tensor
                                Amplitude distribution of the illumination source.
    l2_loss                    : torch.nn.MSELoss
                                L2 loss function.
    loss_type                  : str
                                Type of loss function ("conventional" or "custom").
    loss_function              : callable
                                Loss function to use.
    optimizer                  : torch.optim.Adam
                                Adam optimizer.
    scheduler                  : torch.optim.lr_scheduler.PolynomialLR
                                Learning rate scheduler.
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
        """
        Initialize the multi-color hologram optimizer.

        Parameters
        ------
        wavelengths                : list
                                      List of wavelengths for optimization.
        resolution                 : tuple
                                      Resolution (height, width) of the hologram.
        targets                    : torch.tensor
                                      Target images for optimization.
        propagator                 : propagator
                                      Wave propagation object.
        number_of_frames           : int
                                      Number of temporal frames.
        number_of_depth_layers     : int
                                      Number of depth layers.
        learning_rate              : float
                                      Learning rate for the optimizer.
        scheduler_power            : int
                                      Power for polynomial learning rate scheduler.
        double_phase               : bool
                                      Whether to use double phase encoding.
        scale_factor               : int
                                       Scaling factor for hologram resolution.
                                       method                     : str
                                       Optimization method ("conventional" or "multi-color").
                                       channel_power_filename     : str
                                      Filename to load channel powers from (optional).
        device                     : torch.device
                                      Device to run optimization on.
        loss_function              : callable
                                      Custom loss function (optional).
        peak_amplitude             : float
                                      Peak amplitude for reconstruction.
        optimize_peak_amplitude    : bool
                                      Whether to optimize peak amplitude.
        img_loss_thres             : float
                                      Image loss threshold.
        reduction                  : str
                                      Reduction method for loss ("sum" or "mean").
        """
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
        if self.method not in ["conventional", "multi-color"]:
            raise ValueError(
                f"Unknown optimization method '{self.method}'. Options are 'conventional' or 'multi-color'."
            )
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
        Internal function to initialize the starting peak amplitude as a learnable parameter.

        Parameters
        ------
        None

        Returns
        ------
        None
        """
        self.peak_amplitude = torch.tensor(
            self.peak_amplitude, requires_grad=True, device=self.device
        )

    def init_phase_scale(self):
        """
        Internal function to set the phase scale for each wavelength channel.

        Parameters
        ------
        None

        Returns
        ------
        None
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

        Parameters
        ------
        None

        Returns
        ------
        None
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
        Internal function to initialize the starting phase and offset for phase-only holograms.

        Parameters
        ------
        None

        Returns
        ------
        None
        """
        self.phase = torch.randn(
            self.number_of_frames,
            self.resolution[0],
            self.resolution[1],
            device=self.device,
            requires_grad=False,
        ) * 2. * torch.pi
        self.phase.requires_grad = True
        self.phase_offset = torch.randn(self.number_of_frames, requires_grad=True)

    def init_channel_power(self):
        """
        Internal function to initialize the channel power distribution for multi-color optimization.

        Parameters
        ------
        None

        Returns
        ------
        None
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
        Internal function to set the optimizer with Adam and polynomial learning rate scheduler.

        Parameters
        ------
        number_of_iterations       : int
                                      Total number of optimization iterations.

        Returns
        ------
        None
        """
        optimization_variables = [self.phase, self.phase_offset]
        if self.optimize_peak_amplitude:
            optimization_variables.append(self.peak_amplitude)
        if self.method == "multi-color":
            optimization_variables.append(self.propagator.channel_power)
        self.optimizer = torch.optim.AdamW(optimization_variables, lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
            self.optimizer,
            total_iters=number_of_iterations * 4,
            power=self.scheduler_power,
            last_epoch=-1,
        )

    def init_loss_function(self, loss_function, reduction="sum"):
        """
        Internal function to set the loss function.

        Parameters
        ------
        loss_function              : callable
                                       Custom loss function (optional).
        reduction                  : str
                                       Reduction method for loss ("sum" or "mean").

        Returns
        ------
        None
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

        Parameters
        ------
        input_image                : torch.tensor
                                      Reconstructed image(s).
        target_image               : torch.tensor
                                      Target image(s).
        plane_id                   : int
                                      Depth plane index.
        noise_ratio                : float
                                      Noise ratio for injection.
        inject_noise               : bool
                                      Whether to inject noise into target images.

        Returns
        ------
        loss                       : torch.tensor
                                      Calculated loss value.
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

    def double_phase_constrain(self, phase, phase_offset, levels=6):
        """
        Internal function to constrain a given phase similarly to double phase encoding.

        Parameters
        ------
        phase                      : torch.tensor
                                      Input phase values to be constrained (shape: [1, height]).
        phase_offset               : torch.tensor
                                      Phase offset value.
        levels                     : int
                                      Number of levels for multi-scale total variation loss.

        Returns
        ------
        phase_only                 : torch.tensor
                                      Constrained phase value (shape: [1, height]).
        loss_phase                 : torch.tensor
                                      Total variation loss for constrained phase (scalar).
        """

        phase_zero_mean = phase - torch.mean(phase)
        phase_low, phase_high = decompose_double_phase(phase_zero_mean)
        phase_low = torch.nan_to_num(phase_low - phase_offset, nan=0.0)
        phase_high = torch.nan_to_num(phase_high + phase_offset, nan=torch.pi)
        loss_phase = multi_scale_total_variation_loss(phase_low, levels=levels)
        loss_phase += multi_scale_total_variation_loss(phase_high, levels=levels)
        loss_phase += torch.std(phase_low)
        loss_phase += torch.std(phase_high)
        phase_only = compose_double_phase(phase_high, phase_low)
        return phase_only, loss_phase

    def direct_phase_constrain(self, phase, phase_offset, levels=6):
        """
        Internal function to constrain a given phase.

        Parameters
        ------
        phase                      : torch.tensor
                                      Input phase values to be constrained (shape: [1, height]).
        phase_offset               : torch.tensor
                                      Phase offset value.
        levels                     : int
                                      Number of levels for multi-scale total variation loss.

        Returns
        ------
        phase_only                 : torch.tensor
                                      Constrained output phase (shape: [1, height]).
        loss_phase                 : torch.tensor
                                      Total variation loss for constrained phase (scalar).
        """
        phase_zero_mean = phase - torch.mean(phase)
        phase_only = torch.nan_to_num(phase - phase_offset, nan=torch.pi)
        loss_phase = multi_scale_total_variation_loss(phase_only, levels=levels)
        return phase_only, loss_phase

    def eyebox_constrain(self, phase, offset, diameter):
        """
        Internal function to constrain amplitude distribution in the eyebox region.
        Creates homogeneous amplitude in the masked (eyebox) region and minimal amplitude
        in unmasked regions.

        Parameters
        ------
        phase                      : torch.tensor
                                     Input phase values (shape: [height, width]).
        offset                     : tuple
                                     Offset (x, y) in pixel values for the circular aperture center.
        diameter                   : float
                                     Diameter of the circular aperture in pixel values.

        Returns
        ------
        loss                       : torch.tensor
                                     Combined loss for homogeneous eyebox amplitude and suppression outside.
        """
        complex_field = generate_complex_field(torch.ones_like(phase), phase)
        fft_field = torch.fft.fftshift(torch.fft.fft2(complex_field))
        fft_amplitude = torch.abs(fft_field)
        height, width = fft_amplitude.shape
        radius = diameter / 2.0
        
        mask = circular_binary_mask(height, width, radius, offset_x=offset[0], offset_y=offset[1]).to(self.device).squeeze(0).squeeze(0)
        
        masked_amplitude = fft_amplitude * mask
        unmask = torch.abs(1.0 - mask)
        unmasked_amplitude = fft_amplitude * unmask
        
        masked_amplitude_squeeze = masked_amplitude[mask > 0]
        if masked_amplitude_squeeze.numel() > 0:
            masked_mean = masked_amplitude_squeeze.mean()
            masked_var = masked_amplitude_squeeze.var()
            normalized_var = masked_var / (masked_mean**2 + 1e-8)
        else:
            normalized_var = torch.tensor(1.0, device=self.device)
        
        total_amplitude = fft_amplitude.sum() + 1e-8
        loss_sparsity = unmasked_amplitude.sum() / total_amplitude
        
        loss_homogeneity = normalized_var
        loss_sparsity = loss_sparsity
        
        loss = loss_homogeneity + loss_sparsity
        return loss

    def gradient_descent(
        self,
        number_of_iterations=100,
        weights=None,
        inject_noise=False,
        eyebox={
            "offset": 0.0,
            "diameter" : 100
            },
        noise_ratio=1e-3,
    ):
        """
        Function to optimize multiplane phase-only holograms using stochastic gradient descent.

        Parameters
        ------
        number_of_iterations       : float
                                      Number of iterations.
        weights                    : dict
                                      Loss weights as dictionary with keys: "image", "light", "eyebox", "phase".
        inject_noise               : bool
                                      When set True, this will inject noise with the given `noise_ratio` to the target images.
        eyebox                     : dict
                                      Eyebox constraint parameters with "offset" and "diameter" keys.
        noise_ratio                : float
                                      Noise ratio, a multiplier (1e-3 is 0.1 percent).

        Returns
        -------
        hologram_phases            : torch.tensor
                                     Optimised hologram phases (shape: [number_of_frames, height, width]).
        """
        if weights is None:
            weights = {"image": 1.0, "light": 1.0, "eyebox": 0.0, "phase": 0.0}
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
                loss_light = 0.0    
                loss_eyebox = 0.0
                loss_phase = 0.0
                laser_powers = self.propagator.get_laser_powers()
                for frame_id in range(self.number_of_frames):
                    if weights["phase"] > 0.0:
                        if self.double_phase:
                            phase, loss_phase_frame = self.double_phase_constrain(
                                self.phase[frame_id], self.phase_offset[frame_id]
                            )
                        else:
                            phase, loss_phase_frame = self.direct_phase_constrain(
                                self.phase[frame_id], self.phase_offset[frame_id]
                            )
                        loss_phase += loss_phase_frame
                    else:
                        phase = self.phase[frame_id]
                    if weights["eyebox"] > 0.0:
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
                    if weights["light"] > 0.0:
                        loss_light += self.l2_loss(
                            torch.amax(depth_target, dim=(1, 2)) * self.peak_amplitude,
                            torch.sum(laser_powers, dim=0),
                        )
                        loss_light += self.l2_loss(
                            torch.tensor([self.number_of_frames * self.peak_amplitude]).to(
                                self.device
                            ),
                            torch.sum(laser_powers).view(1,),
                        )
                        loss_light += torch.cos(torch.min(torch.sum(laser_powers, dim=1)))
                reconstruction_intensity = torch.sum(reconstruction_intensities, dim=0)
                loss_image = self.evaluate(
                    reconstruction_intensity,
                    depth_target * self.peak_amplitude,
                    noise_ratio=noise_ratio,
                    inject_noise=inject_noise,
                    plane_id=depth_id,
                )
                loss = weights["image"] * loss_image
                if weights["light"] > 0.0:
                    loss += weights["light"] * loss_light
                if weights["eyebox"] > 0.0:
                    loss += weights["eyebox"] * loss_eyebox
                if weights["phase"] > 0.0:
                    loss += weights["phase"] * loss_phase
                loss.backward(retain_graph=True)
                self.optimizer.step()
                total_loss += loss.detach().item()
                loss_image = loss_image.detach()
                del loss_light
                del loss
            self.scheduler.step()
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
        weights=None,
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
        ------
        number_of_iterations       : int
                                      Number of iterations.
        weights                    : dict
                                      Loss weights as dictionary with keys: "image", "light", "eyebox", "phase".
        eyebox                     : dict
                                      Eyebox constraint parameters.
        bits                       : int
                                      Bit depth for hologram quantization.
        inject_noise               : bool
                                      When set True, this will inject noise with the given `noise_ratio` to the target images.
        noise_ratio                : float
                                      Noise ratio, a multiplier (1e-3 is 0.1 percent).

        Returns
        -------
        hologram_phases            : torch.tensor
                                      Phases of the optimized phase-only hologram.
        reconstruction_intensities : torch.tensor
                                      Intensities of reconstructed images at each plane.
        laser_powers               : torch.tensor
                                      Optimized laser power distribution.
        channel_powers             : torch.tensor
                                      Channel power coefficients.
        peak_amplitude             : float
                                      Final optimized peak amplitude.
        """
        if weights is None:
            weights = {"image": 1.0, "light": 1.0, "eyebox": 1.0, "phase": 0.0}
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
        with torch.no_grad():
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
