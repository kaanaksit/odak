import torch
import copy
from .components import *
from ...log import logger


class multi_layer_perceptron(torch.nn.Module):
    """
    A multi-layer perceptron model.
    """

    def __init__(
        self,
        dimensions,
        activation=torch.nn.ReLU(),
        bias=False,
        model_type="conventional",
        siren_multiplier=30.0,
        input_multiplier=None,
        uniform_flag=True,
    ):
        """
        Initialize the multi-layer perceptron.

        Parameters
        ----------
        dimensions : list of int
            List of integers representing the dimensions of each layer (e.g., [2, 10, 1], where the first element is the input dimension and the last one is the output dimension).
        activation : torch.nn.Module, optional
            Nonlinear activation function. Default is `torch.nn.ReLU()`. Note: this parameter is only utilized when `model_type` is set to 'conventional'.
        bias : bool, optional
            If set to True, linear layers will include biases. Default is False.
        siren_multiplier : float, optional
            When using `SIREN` model type, this parameter functions as a hyperparameter.
            The original SIREN work uses 30.
            You can bypass this parameter by providing input that are not normalized and larger than one. Default is 1.0.
        input_multiplier : float, optional
            Initial value of the input multiplier before the very first layer.
        model_type : str, optional
            Model type: `conventional`, `swish`, `SIREN`, `FILM SIREN`, `Gaussian`.
            `conventional` refers to a standard multi layer perceptron.
            For `SIREN`, see: Sitzmann, Vincent, et al. "Implicit neural representations with periodic activation functions." Advances in neural information processing systems 33 (2020): 7462-7473.
            For `Swish`, see: Ramachandran, Prajit, Barret Zoph, and Quoc V. Le. "Searching for activation functions." arXiv preprint arXiv:1710.05941 (2017).
            For `FILM SIREN`, see: Chan, Eric R., et al. "pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.
            For `Gaussian`, see: Ramasinghe, Sameera, and Simon Lucey. "Beyond periodicity: Towards a unifying framework for activations in coordinate-mlps." In European Conference on Computer Vision, pp. 142-158. Cham: Springer Nature Switzerland, 2022.
            Default is "conventional".
        uniform_flag : bool, optional
            If True, applies specialized uniform initialization for SIREN and FILM SIREN models. Default is True.
        """

        super(multi_layer_perceptron, self).__init__()

        if not isinstance(dimensions, list) or len(dimensions) < 2:
            raise ValueError("dimensions must be a list of at least two integers.")
        if any(not isinstance(d, int) or d <= 0 for d in dimensions):
            raise ValueError("All elements in dimensions must be positive integers.")

        self.bias = bias
        self.model_type = model_type
        self.siren_multiplier = siren_multiplier
        self.dimensions = dimensions
        self.uniform_flag = uniform_flag

        logger.info(
            f"Initializing multi_layer_perceptron: model_type={model_type}, "
            f"dimensions={dimensions}, bias={bias}, "
            f"siren_multiplier={siren_multiplier}, uniform_flag={uniform_flag}"
        )
        
        modules = []
        for i in range(len(self.dimensions) - 1):
            # Add linear layer
            linear_layer = torch.nn.Linear(
                self.dimensions[i], self.dimensions[i + 1], bias=self.bias
            )
            modules.append(linear_layer)

            # Add activation for all but the last layer
            if i < len(self.dimensions) - 2:
                dim = self.dimensions[i + 1]
                if model_type == "conventional":
                    act = copy.deepcopy(activation)
                elif model_type == "swish":
                    act = swish_activation()
                elif model_type == "SIREN":
                    act = siren_activation(multiplier=siren_multiplier)
                elif model_type == "FILM SIREN":
                    act = film_siren_activation(dim=dim)
                elif model_type == "Gaussian":
                    act = gaussian_activation(dim=dim)
                else:
                    raise ValueError(f"Unsupported model_type: {model_type}")
                modules.append(act)

        self.model = torch.nn.Sequential(*modules)

        # SIREN and FILM SIREN specialized weight initialization
        if uniform_flag and model_type in ["SIREN", "FILM SIREN"]:
            with torch.no_grad():
                first_linear = True
                for module in self.model:
                    if isinstance(module, torch.nn.Linear):
                        if first_linear:
                            limit = 1.0 / siren_multiplier if siren_multiplier != 0 else 1.0
                            torch.nn.init.uniform_(module.weight, -limit, limit)
                            first_linear = False
                        else:
                            # Hidden layers scale by omega_0 / sqrt(fan_in) to preserve variance (SIREN paper)
                            fan_in = module.weight.size(1)
                            limit = siren_multiplier / (fan_in ** 0.5) if siren_multiplier != 0 else 1.0 / (fan_in ** 0.5)
                            torch.nn.init.uniform_(module.weight, -limit, limit)

                        if self.bias:
                            torch.nn.init.zeros_(module.bias)

        if input_multiplier is not None:
            self.input_multiplier = torch.nn.Parameter(torch.ones(1, self.dimensions[0]) * input_multiplier)
            logger.debug(f"Input multiplier initialized: {input_multiplier}")

    def forward(self, x):
        """
        Forward pass of the multi-layer perceptron.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (batch_size, input_dim).

        Returns
        -------
        result : torch.Tensor
            Estimated output of shape (batch_size, output_dim).
        """
        if hasattr(self, "input_multiplier"):
            result = x * self.input_multiplier
        else:
            result = x
        return self.model(result)


class unet(torch.nn.Module):
    """
    A U-Net model, heavily inspired from `https://github.com/milesial/Pytorch-UNet/tree/master/unet` and more can be read from Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015.
    """

    def __init__(
        self,
        depth=4,
        dimensions=64,
        input_channels=2,
        output_channels=1,
        bilinear=False,
        kernel_size=3,
        bias=False,
        activation=torch.nn.ReLU(inplace=True),
    ):
        """
        Initialize the U-Net model.

        Parameters
        ----------
        depth : int, optional
            Number of upsampling and downsampling layers. Default is 4.
        dimensions : int, optional
            Number of dimensions. Default is 64.
        input_channels : int, optional
            Number of input channels. Default is 2.
        output_channels : int, optional
            Number of output channels. Default is 1.
        bilinear : bool, optional
            Uses bilinear upsampling in upsampling layers when set True. Default is False.
        kernel_size : int, optional
            Kernel size for convolutional layers. Default is 3.
        bias : bool, optional
            Set True to let convolutional layers learn a bias term. Default is False.
        activation : torch.nn.Module, optional
            Non-linear activation layer to be used (e.g., torch.nn.ReLU(), torch.nn.Sigmoid()). Default is torch.nn.ReLU(inplace=True).
        """
        super(unet, self).__init__()
        logger.info(
            f"Initializing U-Net: depth={depth}, dimensions={dimensions}, "
            f"input_channels={input_channels}, output_channels={output_channels}, "
            f"bilinear={bilinear}, kernel_size={kernel_size}"
        )
        self.inc = double_convolution(
            input_channels=input_channels,
            mid_channels=dimensions,
            output_channels=dimensions,
            kernel_size=kernel_size,
            bias=bias,
            activation=activation,
        )

        self.downsampling_layers = torch.nn.ModuleList()
        self.upsampling_layers = torch.nn.ModuleList()
        for i in range(depth):  # downsampling layers
            in_channels = dimensions * (2**i)
            out_channels = dimensions * (2 ** (i + 1))
            down_layer = downsample_layer(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                activation=activation,
            )
            self.downsampling_layers.append(down_layer)
            logger.debug(f"Added downsampling layer {i}: {in_channels} -> {out_channels}")

        for i in range(depth - 1, -1, -1):  # upsampling layers
            up_in_channels = dimensions * (2 ** (i + 1))
            up_out_channels = dimensions * (2**i)
            up_layer = upsample_layer(
                up_in_channels,
                up_out_channels,
                kernel_size=kernel_size,
                bias=bias,
                activation=activation,
                bilinear=bilinear,
            )
            self.upsampling_layers.append(up_layer)
            logger.debug(f"Added upsampling layer: {up_in_channels} -> {up_out_channels}")
        self.outc = torch.nn.Conv2d(
            dimensions,
            output_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )
        logger.info("U-Net initialization completed")

    def forward(self, x):
        """
        Forward pass of the U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        result : torch.Tensor
            Estimated output.
        """
        downsampling_outputs = [self.inc(x)]
        for down_layer in self.downsampling_layers:
            x_down = down_layer(downsampling_outputs[-1])
            downsampling_outputs.append(x_down)
        x_up = downsampling_outputs[-1]
        for i, up_layer in enumerate((self.upsampling_layers)):
            x_up = up_layer(x_up, downsampling_outputs[-(i + 2)])
        result = self.outc(x_up)
        return result


class spatially_varying_kernel_generation_model(torch.nn.Module):
    """
    Spatially_varying_kernel_generation_model revised from RSGUnet:
    https://github.com/MTLab/rsgunet_image_enhance.

    Refer to:
    J. Huang, P. Zhu, M. Geng et al. Range Scaling Global U-Net for Perceptual Image Enhancement on Mobile Devices.
    """

    def __init__(
        self,
        depth=3,
        dimensions=8,
        input_channels=7,
        kernel_size=3,
        bias=True,
        normalization=False,
        activation=torch.nn.LeakyReLU(0.2, inplace=True),
    ):
        """
        Initialize the spatially varying kernel generation model.

        Parameters
        ----------
        depth : int, optional
            Number of upsampling and downsampling layers. Default is 3.
        dimensions : int, optional
            Number of dimensions. Default is 8.
        input_channels : int, optional
            Number of input channels. Default is 7.
        kernel_size : int, optional
            Kernel size for convolutional layers. Default is 3.
        bias : bool, optional
            Set to True to let convolutional layers learn a bias term. Default is True.
        normalization : bool, optional
            If True, adds a Batch Normalization layer after the convolutional layer. Default is False.
        activation : torch.nn.Module, optional
            Non-linear activation layer (e.g., torch.nn.ReLU(), torch.nn.Sigmoid()). Default is torch.nn.LeakyReLU(0.2, inplace=True).
        """
        super().__init__()
        self.depth = depth
        logger.info(
            f"Initializing spatially_varying_kernel_generation_model: "
            f"depth={depth}, dimensions={dimensions}, input_channels={input_channels}, "
            f"kernel_size={kernel_size}, bias={bias}, normalization={normalization}"
        )
        self.inc = convolution_layer(
            input_channels=input_channels,
            output_channels=dimensions,
            kernel_size=kernel_size,
            bias=bias,
            normalization=normalization,
            activation=activation,
        )

        self.encoder = torch.nn.ModuleList()
        for i in range(depth + 1):  # downsampling layers
            if i == 0:
                in_channels = dimensions * (2**i)
                out_channels = dimensions * (2**i)
            elif i == depth:
                in_channels = dimensions * (2 ** (i - 1))
                out_channels = dimensions * (2 ** (i - 1))
            else:
                in_channels = dimensions * (2 ** (i - 1))
                out_channels = 2 * in_channels
            pooling_layer = torch.nn.AvgPool2d(2)
            double_convolution_layer = double_convolution(
                input_channels=in_channels,
                mid_channels=in_channels,
                output_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
                normalization=normalization,
                activation=activation,
            )
            self.encoder.append(pooling_layer)
            self.encoder.append(double_convolution_layer)
            logger.debug(f"Added encoder block {i}: {in_channels} -> {out_channels}")

        self.spatially_varying_feature = torch.nn.ModuleList()  # for kernel generation
        for i in range(depth, -1, -1):
            if i == 1:
                svf_in_channels = dimensions + 2 ** (self.depth + i) + 1
            else:
                svf_in_channels = 2 ** (self.depth + i) + 1
            svf_out_channels = (2 ** (self.depth + i)) * (kernel_size * kernel_size)
            svf_mid_channels = dimensions * (2 ** (self.depth - 1))
            spatially_varying_kernel_generation = torch.nn.ModuleList()
            for j in range(i, -1, -1):
                pooling_layer = torch.nn.AvgPool2d(2 ** (j + 1))
                spatially_varying_kernel_generation.append(pooling_layer)
            kernel_generation_block = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=svf_in_channels,
                    out_channels=svf_mid_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=bias,
                ),
                activation,
                torch.nn.Conv2d(
                    in_channels=svf_mid_channels,
                    out_channels=svf_mid_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=bias,
                ),
                activation,
                torch.nn.Conv2d(
                    in_channels=svf_mid_channels,
                    out_channels=svf_out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=bias,
                ),
            )
            spatially_varying_kernel_generation.append(kernel_generation_block)
            self.spatially_varying_feature.append(spatially_varying_kernel_generation)
            logger.debug(f"Added SVF block {i}: {svf_in_channels} -> {svf_out_channels}")

        self.decoder = torch.nn.ModuleList()
        global_feature_layer = global_feature_module(  # global feature layer
            input_channels=dimensions * (2 ** (depth - 1)),
            mid_channels=dimensions * (2 ** (depth - 1)),
            output_channels=dimensions * (2 ** (depth - 1)),
            kernel_size=kernel_size,
            bias=bias,
            activation=torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder.append(global_feature_layer)
        for i in range(depth, 0, -1):
            if i == 2:
                up_in_channels = (dimensions // 2) * (2**i)
                up_out_channels = up_in_channels
                up_mid_channels = up_in_channels
            elif i == 1:
                up_in_channels = dimensions * 2
                up_out_channels = dimensions
                up_mid_channels = up_out_channels
            else:
                up_in_channels = (dimensions // 2) * (2**i)
                up_out_channels = up_in_channels // 2
                up_mid_channels = up_in_channels
            upsample_layer = upsample_convtranspose2d_layer(
                input_channels=up_in_channels,
                output_channels=up_mid_channels,
                kernel_size=2,
                stride=2,
                bias=bias,
            )
            conv_layer = double_convolution(
                input_channels=up_mid_channels,
                output_channels=up_out_channels,
                kernel_size=kernel_size,
                bias=bias,
                normalization=normalization,
                activation=activation,
            )
            self.decoder.append(torch.nn.ModuleList([upsample_layer, conv_layer]))
            logger.debug(f"Added decoder block {i}: {up_in_channels} -> {up_out_channels}")
        logger.info("spatially_varying_kernel_generation_model initialization completed")

    def forward(self, focal_surface, field):
        """
        Forward pass of the spatially varying kernel generation model.

        Parameters
        ----------
        focal_surface : torch.Tensor
            Input focal surface data.
            Dimension: (1, 1, H, W)

        field : torch.Tensor
            Input field data.
            Dimension: (1, 6, H, W)

        Returns
        -------
        sv_kernel : list of torch.Tensor
            Learned spatially varying kernels.
            Dimension of each element in the list: (1, C_i * kernel_size * kernel_size, H_i, W_i),
            where C_i, H_i, and W_i represent the channel, height, and width
            of each feature at a certain scale.
        """
        x = self.inc(torch.cat((focal_surface, field), dim=1))
        downsampling_outputs = [focal_surface]
        downsampling_outputs.append(x)
        for i, down_layer in enumerate(self.encoder):
            x_down = down_layer(downsampling_outputs[-1])
            downsampling_outputs.append(x_down)
        sv_kernels = []
        for i, (up_layer, svf_layer) in enumerate(
            zip(self.decoder, self.spatially_varying_feature)
        ):
            if i == 0:
                global_feature = up_layer(
                    downsampling_outputs[-2], downsampling_outputs[-1]
                )
                downsampling_outputs[-1] = global_feature
                sv_feature = [global_feature, downsampling_outputs[0]]
                for j in range(self.depth - i + 1):
                    sv_feature[1] = svf_layer[self.depth - i](sv_feature[1])
                    if j > 0:
                        sv_feature.append(svf_layer[j](downsampling_outputs[2 * j]))
                sv_feature = [
                    sv_feature[0],
                    sv_feature[1],
                    sv_feature[4],
                    sv_feature[2],
                    sv_feature[3],
                ]
                sv_kernel = svf_layer[-1](torch.cat(sv_feature, dim=1))
                sv_kernels.append(sv_kernel)
            else:
                x_up = up_layer[0](
                    downsampling_outputs[-1],
                    downsampling_outputs[2 * (self.depth + 1 - i) + 1],
                )
                x_up = up_layer[1](x_up)
                downsampling_outputs[-1] = x_up
                sv_feature = [x_up, downsampling_outputs[0]]
                for j in range(self.depth - i + 1):
                    sv_feature[1] = svf_layer[self.depth - i](sv_feature[1])
                    if j > 0:
                        sv_feature.append(svf_layer[j](downsampling_outputs[2 * j]))
                if i == 1:
                    sv_feature = [
                        sv_feature[0],
                        sv_feature[1],
                        sv_feature[3],
                        sv_feature[2],
                    ]
                sv_kernel = svf_layer[-1](torch.cat(sv_feature, dim=1))
                sv_kernels.append(sv_kernel)
        return sv_kernels


class spatially_adaptive_unet(torch.nn.Module):
    """
    Spatially varying U-Net model based on spatially adaptive convolution.

    References
    ----------
    Chuanjun Zheng, Yicheng Zhan, Liang Shi, Ozan Cakmakci, and Kaan Akşit, "Focal Surface Holographic Light Transport using Learned Spatially Adaptive Convolutions," SIGGRAPH Asia 2024 Technical Communications (SA Technical Communications '24), December, 2024.
    """

    def __init__(
        self,
        depth=3,
        dimensions=8,
        input_channels=6,
        out_channels=6,
        kernel_size=3,
        bias=True,
        normalization=False,
        activation=torch.nn.LeakyReLU(0.2, inplace=True),
    ):
        """
        Initialize the spatially adaptive U-Net model.

        Parameters
        ----------
        depth : int, optional
            Number of upsampling and downsampling layers. Default is 3.
        dimensions : int, optional
            Number of dimensions. Default is 8.
        input_channels : int, optional
            Number of input channels. Default is 6.
        out_channels : int, optional
            Number of output channels. Default is 6.
        kernel_size : int, optional
            Kernel size for convolutional layers. Default is 3.
        bias : bool, optional
            Set to True to let convolutional layers learn a bias term. Default is True.
        normalization : bool, optional
            If True, adds a Batch Normalization layer after the convolutional layer. Default is False.
        activation : torch.nn.Module, optional
            Non-linear activation layer (e.g., torch.nn.ReLU(), torch.nn.Sigmoid()). Default is torch.nn.LeakyReLU(0.2, inplace=True).
        """
        super().__init__()
        self.depth = depth
        self.out_channels = out_channels
        logger.info(
            f"Initializing spatially_adaptive_unet: "
            f"depth={depth}, dimensions={dimensions}, input_channels={input_channels}, "
            f"out_channels={out_channels}, kernel_size={kernel_size}, "
            f"bias={bias}, normalization={normalization}"
        )
        self.inc = convolution_layer(
            input_channels=input_channels,
            output_channels=dimensions,
            kernel_size=kernel_size,
            bias=bias,
            normalization=normalization,
            activation=activation,
        )

        self.encoder = torch.nn.ModuleList()
        for i in range(self.depth + 1):  # Downsampling layers
            down_in_channels = dimensions * (2**i)
            down_out_channels = 2 * down_in_channels
            pooling_layer = torch.nn.AvgPool2d(2)
            double_convolution_layer = double_convolution(
                input_channels=down_in_channels,
                mid_channels=down_in_channels,
                output_channels=down_in_channels,
                kernel_size=kernel_size,
                bias=bias,
                normalization=normalization,
                activation=activation,
            )
            sam = spatially_adaptive_module(
                input_channels=down_in_channels,
                output_channels=down_out_channels,
                kernel_size=kernel_size,
                bias=bias,
                activation=activation,
            )
            self.encoder.append(
                torch.nn.ModuleList([pooling_layer, double_convolution_layer, sam])
            )
            logger.debug(f"Added encoder block {i}: {down_in_channels} -> {down_out_channels}")
        self.global_feature_module = torch.nn.ModuleList()
        double_convolution_layer = double_convolution(
            input_channels=dimensions * (2 ** (depth + 1)),
            mid_channels=dimensions * (2 ** (depth + 1)),
            output_channels=dimensions * (2 ** (depth + 1)),
            kernel_size=kernel_size,
            bias=bias,
            normalization=normalization,
            activation=activation,
        )
        global_feature_layer = global_feature_module(
            input_channels=dimensions * (2 ** (depth + 1)),
            mid_channels=dimensions * (2 ** (depth + 1)),
            output_channels=dimensions * (2 ** (depth + 1)),
            kernel_size=kernel_size,
            bias=bias,
            activation=torch.nn.LeakyReLU(0.2, inplace=True),
        )
        self.global_feature_module.append(
            torch.nn.ModuleList([double_convolution_layer, global_feature_layer])
        )
        logger.debug("Added global feature module")

        self.decoder = torch.nn.ModuleList()
        for i in range(depth, -1, -1):
            up_in_channels = dimensions * (2 ** (i + 1))
            up_mid_channels = up_in_channels // 2
            if i == 0:
                up_out_channels = self.out_channels
                upsample_layer = upsample_convtranspose2d_layer(
                    input_channels=up_in_channels,
                    output_channels=up_mid_channels,
                    kernel_size=2,
                    stride=2,
                    bias=bias,
                )
                conv_layer = torch.nn.Sequential(
                    convolution_layer(
                        input_channels=up_mid_channels,
                        output_channels=up_mid_channels,
                        kernel_size=kernel_size,
                        bias=bias,
                        normalization=normalization,
                        activation=activation,
                    ),
                    convolution_layer(
                        input_channels=up_mid_channels,
                        output_channels=up_out_channels,
                        kernel_size=1,
                        bias=bias,
                        normalization=normalization,
                        activation=None,
                    ),
                )
                self.decoder.append(torch.nn.ModuleList([upsample_layer, conv_layer]))
                logger.debug(f"Added decoder block {i}: {up_in_channels} -> {up_out_channels}")
            else:
                up_out_channels = up_in_channels // 2
                upsample_layer = upsample_convtranspose2d_layer(
                    input_channels=up_in_channels,
                    output_channels=up_mid_channels,
                    kernel_size=2,
                    stride=2,
                    bias=bias,
                )
                conv_layer = double_convolution(
                    input_channels=up_mid_channels,
                    mid_channels=up_mid_channels,
                    output_channels=up_out_channels,
                    kernel_size=kernel_size,
                    bias=bias,
                    normalization=normalization,
                    activation=activation,
                )
                self.decoder.append(torch.nn.ModuleList([upsample_layer, conv_layer]))
                logger.debug(f"Added decoder block {i}: {up_in_channels} -> {up_out_channels}")
        logger.info("spatially_adaptive_unet initialization completed")

    def forward(self, sv_kernel, field):
        """
        Forward pass of the spatially adaptive U-Net.

        Parameters
        ----------
        sv_kernel : list of torch.Tensor
            Learned spatially varying kernels.
            Dimension of each element in the list: (1, C_i * kernel_size * kernel_size, H_i, W_i),
            where C_i, H_i, and W_i represent the channel, height, and width
            of each feature at a certain scale.

        field : torch.Tensor
            Input field data.
            Dimension: (1, 6, H, W)

        Returns
        -------
        target_field : torch.Tensor
            Estimated output.
            Dimension: (1, 6, H, W)
        """
        x = self.inc(field)
        downsampling_outputs = [x]
        for i, down_layer in enumerate(self.encoder):
            x_down = down_layer[0](downsampling_outputs[-1])
            downsampling_outputs.append(x_down)
            sam_output = down_layer[2](
                x_down + down_layer[1](x_down), sv_kernel[self.depth - i]
            )
            downsampling_outputs.append(sam_output)
        global_feature = self.global_feature_module[0][0](downsampling_outputs[-1])
        global_feature = self.global_feature_module[0][1](
            downsampling_outputs[-1], global_feature
        )
        downsampling_outputs.append(global_feature)
        x_up = downsampling_outputs[-1]
        for i, up_layer in enumerate(self.decoder):
            x_up = up_layer[0](x_up, downsampling_outputs[2 * (self.depth - i)])
            x_up = up_layer[1](x_up)
        result = x_up
        return result
