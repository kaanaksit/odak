import torch
import os
import json
import numpy as np
from tqdm import tqdm
from ..models import *
from .util import generate_complex_field, wavenumber, calculate_amplitude
from ...tools.file import validate_path, check_directory
from ...log import logger
from os.path import join


class holobeam_multiholo(torch.nn.Module):
    """
    The learned holography model used in the paper, Akşit, Kaan, and Yuta Itoh. "HoloBeam: Paper-Thin Near-Eye Displays." In 2023 IEEE Conference Virtual Reality and 3D User Interfaces (VR), pp. 581-591. IEEE, 2023.


    Parameters
    ----------
    n_input           : int
                        Number of channels in the input.
    n_hidden          : int
                        Number of channels in the hidden layers.
    n_output          : int
                        Number of channels in the output layer.
    device            : torch.device
                        Default device is CPU.
    reduction         : str
                        Reduction used for torch.nn.MSELoss and torch.nn.L1Loss. The default is 'sum'.
    """

    def __init__(
        self,
        n_input=1,
        n_hidden=16,
        n_output=2,
        device=torch.device("cpu"),
        reduction="sum",
    ):
        super(holobeam_multiholo, self).__init__()
        torch.random.seed()
        self.device = device
        self.reduction = reduction
        self.l2 = torch.nn.MSELoss(reduction=self.reduction)
        self.l1 = torch.nn.L1Loss(reduction=self.reduction)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        logger.info(
            f"Initializing holobeam_multiholo: n_input={n_input}, n_hidden={n_hidden}, "
            f"n_output={n_output}, device={device}, reduction={reduction}"
        )
        self.network = unet(
            dimensions=self.n_hidden,
            input_channels=self.n_input,
            output_channels=self.n_output,
        ).to(self.device)

    def forward(self, x, test=False):
        """
        Internal function representing the forward model.
        """
        if test:
            torch.no_grad()
        y = self.network.forward(x)
        phase_low = y[:, 0].unsqueeze(1)
        phase_high = y[:, 1].unsqueeze(1)
        phase_only = torch.zeros_like(phase_low)
        phase_only[:, :, 0::2, 0::2] = phase_low[:, :, 0::2, 0::2]
        phase_only[:, :, 1::2, 1::2] = phase_low[:, :, 1::2, 1::2]
        phase_only[:, :, 0::2, 1::2] = phase_high[:, :, 0::2, 1::2]
        phase_only[:, :, 1::2, 0::2] = phase_high[:, :, 1::2, 0::2]
        return phase_only

    def evaluate(self, input_data, ground_truth, weights=[1.0, 0.1]):
        """
        Internal function for evaluating.
        """
        loss = weights[0] * self.l2(input_data, ground_truth) + weights[1] * self.l1(
            input_data, ground_truth
        )
        return loss

    def fit(
        self,
        dataloader,
        number_of_epochs=100,
        learning_rate=1e-5,
        directory="./output",
        save_at_every=100,
    ):
        """
        Function to train the weights of the multi layer perceptron.

        Parameters
        ----------
        dataloader       : torch.utils.data.DataLoader
                           Data loader.
        number_of_epochs : int
                           Number of epochs.
        learning_rate    : float
                           Learning rate of the optimizer.
        directory        : str
                            Output directory.
        save_at_every    : int
                            Save the model at every epoch count.

        Raises
        ------
        ValueError    : If directory path contains dangerous patterns (traversal, null bytes, etc.).
        TypeError     : If directory is not a string.
        """
        safe_directory = validate_path(directory)
        check_directory(safe_directory, validate=True)
        logger.info(
            f"Starting training: epochs={number_of_epochs}, lr={learning_rate}, "
            f"directory={safe_directory}, save_interval={save_at_every}"
        )
        t_epoch = tqdm(range(number_of_epochs), leave=False, dynamic_ncols=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for i in t_epoch:
            epoch_loss = 0.0
            t_data = tqdm(dataloader, leave=False, dynamic_ncols=True)
            for j, data in enumerate(t_data):
                self.optimizer.zero_grad()
                images, holograms = data
                images = images.to(self.device, non_blocking=True)
                holograms = holograms.to(self.device, non_blocking=True)
                estimates = self.forward(images)
                loss = self.evaluate(estimates, holograms)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                description = "Loss:{:.4f}".format(loss.item())
                t_data.set_description(description)
                epoch_loss += float(loss.item()) / dataloader.__len__()
            description = "Epoch Loss:{:.4f}".format(epoch_loss)
            t_epoch.set_description(description)
            if i % save_at_every == 0:
                weight_file = join(safe_directory, f"weights_{i:04d}.pt")
                self.save_weights(filename=weight_file)
                logger.info(f"Model saved at epoch {i}: {weight_file}")
        final_weight_file = join(safe_directory, "weights.pt")
        self.save_weights(filename=final_weight_file)
        logger.info(f"Training completed. Final model saved: {final_weight_file}")
        logger.warning(description)

    def save_weights(self, filename="./weights.pt"):
        """
        Function to save the current weights of the multi layer perceptron to a file.

        Parameters
        ----------
        filename        : str
                          Filename.

        Raises
        ------
        ValueError      : If path validation fails or extension is not allowed.
        TypeError       : If filename is not a string.
        """
        safe_path = validate_path(filename, allowed_extensions=[".pt", ".pth"])
        torch.save(self.network.state_dict(), safe_path)
        logger.debug(f"Model weights saved: {safe_path}")

    def load_weights(self, filename="./weights.pt"):
        """
        Function to load weights for this multi layer perceptron from a file.

        Parameters
        ----------
        filename        : str
                          Filename.

        Raises
        ------
        ValueError      : If path validation fails or extension is not allowed.
        TypeError       : If filename is not a string.
        """
        safe_path = validate_path(filename, allowed_extensions=[".pt", ".pth"])
        self.network.load_state_dict(torch.load(safe_path, weights_only=True))
        self.network.eval()
        logger.info(f"Model weights loaded from: {safe_path}")


class focal_surface_light_propagation(torch.nn.Module):
    """
    focal_surface_light_propagation model.

    References
    ----------

    Chuanjun Zheng, Yicheng Zhan, Liang Shi, Ozan Cakmakci, and Kaan Akşit}. "Focal Surface Holographic Light Transport using Learned Spatially Adaptive Convolutions." SIGGRAPH Asia 2024 Technical Communications (SA Technical Communications '24),December,2024.
    """

    def __init__(
        self,
        depth=3,
        dimensions=8,
        input_channels=6,
        out_channels=6,
        kernel_size=3,
        bias=True,
        device=torch.device("cpu"),
        activation=torch.nn.LeakyReLU(0.2, inplace=True),
    ):
        """
        Initializes the focal surface light propagation model.

        Parameters
        ----------
        depth             : int
                            Number of downsampling and upsampling layers.
        dimensions        : int
                            Number of dimensions/features in the model.
        input_channels    : int
                            Number of input channels.
        out_channels      : int
                            Number of output channels.
        kernel_size       : int
                            Size of the convolution kernel.
        bias              : bool
                            If True, allows convolutional layers to learn a bias term.
        device            : torch.device
                            Default device is CPU.
        activation        : torch.nn.Module
                            Activation function (e.g., torch.nn.ReLU(), torch.nn.Sigmoid()).
        """
        super().__init__()
        self.depth = depth
        self.device = device
        logger.info(
            f"Initializing focal_surface_light_propagation: depth={depth}, "
            f"dimensions={dimensions}, input_channels={input_channels}, "
            f"out_channels={out_channels}, kernel_size={kernel_size}, device={device}"
        )
        self.sv_kernel_generation = spatially_varying_kernel_generation_model(
            depth=depth,
            dimensions=dimensions,
            input_channels=input_channels + 1,  # +1 to account for an extra channel
            kernel_size=kernel_size,
            bias=bias,
            activation=activation,
        )
        self.light_propagation = spatially_adaptive_unet(
            depth=depth,
            dimensions=dimensions,
            input_channels=input_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            activation=activation,
        )

    def forward(self, focal_surface, phase_only_hologram):
        """
        Forward pass through the model.

        Parameters
        ----------
        focal_surface         : torch.Tensor
                                Input focal surface.
        phase_only_hologram   : torch.Tensor
                                Input phase-only hologram.

        Returns
        ----------
        result                : torch.Tensor
                                Output tensor after light propagation.
        """
        input_field = self.generate_input_field(phase_only_hologram)
        sv_kernel = self.sv_kernel_generation(focal_surface, input_field)
        output_field = self.light_propagation(sv_kernel, input_field)
        final = output_field[:, 0:3, :, :] + 1j * output_field[:, 3:6, :, :]
        result = calculate_amplitude(final) ** 2
        return result

    def generate_input_field(self, phase_only_hologram):
        """
        Generates an input field by combining the real and imaginary parts.

        Parameters
        ----------
        phase_only_hologram   : torch.Tensor
                                Input phase-only hologram.

        Returns
        ----------
        input_field           : torch.Tensor
                                Concatenated real and imaginary parts of the complex field.
        """
        [b, c, h, w] = phase_only_hologram.size()
        input_phase = phase_only_hologram * 2 * np.pi
        hologram_amplitude = torch.ones(
            b, c, h, w, requires_grad=False, device=phase_only_hologram.device
        )
        field = generate_complex_field(hologram_amplitude, input_phase)
        input_field = torch.cat((field.real, field.imag), dim=1)
        return input_field

    def load_weights(self, weight_filename, key_mapping_filename):
        """
        Function to load weights for this multi-layer perceptron from a file.

        Parameters
        ----------
        weight_filename      : str
                               Path to the old model's weight file.
        key_mapping_filename : str
                               Path to the JSON file containing the key mappings.

        Raises
        ------
        ValueError      : If path validation fails or extension is not allowed.
        TypeError       : If filenames are not strings.
        """
        logger.info(
            f"Loading weights from {weight_filename} with key mapping {key_mapping_filename}"
        )
        # Validate and load old model weights
        safe_weight_path = validate_path(
            weight_filename, allowed_extensions=[".pt", ".pth"]
        )
        old_model_weights = torch.load(
            safe_weight_path, map_location=self.device, weights_only=True
        )

        # Validate and load key mappings from JSON file
        safe_key_map_path = validate_path(
            key_mapping_filename, allowed_extensions=[".json"]
        )
        with open(safe_key_map_path, "r") as json_file:
            key_mappings = json.load(json_file)

        # Extract the key mappings for sv_kernel_generation and light_prop
        sv_kernel_generation_key_mapping = key_mappings[
            "sv_kernel_generation_key_mapping"
        ]
        light_prop_key_mapping = key_mappings["light_prop_key_mapping"]

        # Initialize new state dicts
        sv_kernel_generation_new_state_dict = {}
        light_prop_new_state_dict = {}

        # Map and load sv_kernel_generation_model weights
        for old_key, value in old_model_weights.items():
            if old_key in sv_kernel_generation_key_mapping:
                # Map the old key to the new key
                new_key = sv_kernel_generation_key_mapping[old_key]
                sv_kernel_generation_new_state_dict[new_key] = value

        self.sv_kernel_generation.to(self.device)
        self.sv_kernel_generation.load_state_dict(sv_kernel_generation_new_state_dict)
        logger.debug(
            f"Loaded {len(sv_kernel_generation_new_state_dict)} keys for sv_kernel_generation"
        )

        # Map and load light_prop model weights
        for old_key, value in old_model_weights.items():
            if old_key in light_prop_key_mapping:
                # Map the old key to the new key
                new_key = light_prop_key_mapping[old_key]
                light_prop_new_state_dict[new_key] = value
        self.light_propagation.to(self.device)
        self.light_propagation.load_state_dict(light_prop_new_state_dict)
        logger.debug(
            f"Loaded {len(light_prop_new_state_dict)} keys for light_propagation"
        )
        logger.info("Weights successfully loaded and mapped to both sub-models")
