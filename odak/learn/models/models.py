import torch
from .components import *


class multi_layer_perceptron(torch.nn.Module):
    """
    A multi-layer perceptron model.
    """

    def __init__(self,
                 dimensions,
                 activation = torch.nn.ReLU(),
                 bias = False,
                 model_type = 'conventional',
                 siren_multiplier = 1.,
                 input_multiplier = None
                ):
        """
        Parameters
        ----------
        dimensions        : list
                            List of integers representing the dimensions of each layer (e.g., [2, 10, 1], where the first layer has two channels and last one has one channel.).
        activation        : torch.nn
                            Nonlinear activation function.
                            Default is `torch.nn.ReLU()`.
        bias              : bool
                            If set to True, linear layers will include biases.
        siren_multiplier  : float
                            When using `SIREN` model type, this parameter functions as a hyperparameter.
                            The original SIREN work uses 30.
                            You can bypass this parameter by providing input that are not normalized and larger then one.
        input_multiplier  : float
                            Initial value of the input multiplier before the very first layer.
        model_type        : str
                            Model type: `conventional`, `swish`, `SIREN`, `FILM SIREN`, `Gaussian`.
                            `conventional` refers to a standard multi layer perceptron.
                            For `SIREN,` see: Sitzmann, Vincent, et al. "Implicit neural representations with periodic activation functions." Advances in neural information processing systems 33 (2020): 7462-7473.
                            For `Swish,` see: Ramachandran, Prajit, Barret Zoph, and Quoc V. Le. "Searching for activation functions." arXiv preprint arXiv:1710.05941 (2017). 
                            For `FILM SIREN,` see: Chan, Eric R., et al. "pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.
                            For `Gaussian,` see: Ramasinghe, Sameera, and Simon Lucey. "Beyond periodicity: Towards a unifying framework for activations in coordinate-mlps." In European Conference on Computer Vision, pp. 142-158. Cham: Springer Nature Switzerland, 2022.
        """
        super(multi_layer_perceptron, self).__init__()
        self.activation = activation
        self.bias = bias
        self.model_type = model_type
        self.layers = torch.nn.ModuleList()
        self.siren_multiplier = siren_multiplier
        self.dimensions = dimensions
        for i in range(len(self.dimensions) - 1):
            self.layers.append(torch.nn.Linear(self.dimensions[i], self.dimensions[i + 1], bias = self.bias))
        if not isinstance(input_multiplier, type(None)):
            self.input_multiplier = torch.nn.ParameterList()
            self.input_multiplier.append(torch.nn.Parameter(torch.ones(1, self.dimensions[0]) * input_multiplier))
        if self.model_type == 'FILM SIREN':
            self.alpha = torch.nn.ParameterList()
            for j in self.dimensions[1:-1]:
                self.alpha.append(torch.nn.Parameter(torch.randn(2, 1, j)))
        if self.model_type == 'Gaussian':
            self.alpha = torch.nn.ParameterList()
            for j in self.dimensions[1:-1]:
                self.alpha.append(torch.nn.Parameter(torch.randn(1, 1, j)))


    def forward(self, x):
        """
        Forward model.
        
        Parameters
        ----------
        x             : torch.tensor
                        Input data.
      
 
        Returns
        ----------
        result        : torch.tensor
                        Estimated output.      
        """
        if hasattr(self, 'input_multiplier'):
            result = x * self.input_multiplier[0]
        else:
            result = x
        for layer_id, layer in enumerate(self.layers[:-1]):
            result = layer(result)
            if self.model_type == 'conventional':
                result = self.activation(result)
            elif self.model_type == 'swish':
                resutl = swish(result)
            elif self.model_type == 'SIREN':
                result = torch.sin(result * self.siren_multiplier)
            elif self.model_type == 'FILM SIREN':
                result = torch.sin(self.alpha[layer_id][0] * result + self.alpha[layer_id][1])
            elif self.model_type == 'Gaussian': 
                result = gaussian(result, self.alpha[layer_id][0])
        result = self.layers[-1](result)
        return result


class unet(torch.nn.Module):
    """
    A U-Net model, heavily inspired from `https://github.com/milesial/Pytorch-UNet/tree/master/unet` and more can be read from Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015.
    """

    def __init__(
                 self, 
                 depth = 4,
                 dimensions = 64, 
                 input_channels = 2, 
                 output_channels = 1, 
                 bilinear = False,
                 kernel_size = 3,
                 bias = False,
                 activation = torch.nn.ReLU(inplace = True),
                ):
        """
        U-Net model.

        Parameters
        ----------
        depth             : int
                            Number of upsampling and downsampling
        dimensions        : int
                            Number of dimensions.
        input_channels    : int
                            Number of input channels.
        output_channels   : int
                            Number of output channels.
        bilinear          : bool
                            Uses bilinear upsampling in upsampling layers when set True.
        bias              : bool
                            Set True to let convolutional layers learn a bias term.
        activation        : torch.nn
                            Non-linear activation layer to be used (e.g., torch.nn.ReLU(), torch.nn.Sigmoid().
        """
        super(unet, self).__init__()
        self.inc = double_convolution(
                                      input_channels = input_channels,
                                      mid_channels = dimensions,
                                      output_channels = dimensions,
                                      kernel_size = kernel_size,
                                      bias = bias,
                                      activation = activation
                                     )      
        
        self.downsampling_layers = torch.nn.ModuleList()
        self.upsampling_layers = torch.nn.ModuleList()
        for i in range(depth): # downsampling layers
            in_channels = dimensions * (2 ** i)
            out_channels = dimensions * (2 ** (i + 1))
            down_layer = downsample_layer(in_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            bias=bias,
                                            activation=activation
                                            )
            self.downsampling_layers.append(down_layer)      
       
        for i in range(depth - 1, -1, -1):  # upsampling layers
            up_in_channels = dimensions * (2 ** (i + 1))  
            up_out_channels = dimensions * (2 ** i) 
            up_layer = upsample_layer(up_in_channels, up_out_channels, kernel_size=kernel_size, bias=bias, activation=activation, bilinear=bilinear)
            self.upsampling_layers.append(up_layer)
        self.outc = torch.nn.Conv2d(
                                    dimensions, 
                                    output_channels,
                                    kernel_size = kernel_size,
                                    padding = kernel_size // 2,
                                    bias = bias
                                   )

    
    def forward(self, x):
        """
        Forward model.
        
        Parameters
        ----------
        x             : torch.tensor
                        Input data.
      
 
        Returns
        ----------
        result        : torch.tensor
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
                 depth = 3,
                 dimensions = 8,
                 input_channels = 7,
                 kernel_size = 3,
                 bias = True,
                 normalization = False,
                 activation = torch.nn.LeakyReLU(0.2, inplace = True)
                ):
        """
        U-Net model.

        Parameters
        ----------
        depth          : int
                         Number of upsampling and downsampling layers.
        dimensions     : int
                         Number of dimensions.
        input_channels : int
                         Number of input channels.
        bias           : bool
                         Set to True to let convolutional layers learn a bias term.
        normalization  : bool
                         If True, adds a Batch Normalization layer after the convolutional layer.
        activation     : torch.nn
                         Non-linear activation layer (e.g., torch.nn.ReLU(), torch.nn.Sigmoid()).
        """
        super().__init__()
        self.depth = depth
        self.inc = convolution_layer(
                                     input_channels = input_channels,
                                     output_channels = dimensions,
                                     kernel_size = kernel_size,
                                     bias = bias,
                                     normalization = normalization,
                                     activation = activation
                                    )
        self.encoder = torch.nn.ModuleList()
        for i in range(depth + 1):  # downsampling layers
            if i == 0:
                in_channels = dimensions * (2 ** i)
                out_channels = dimensions * (2 ** i)
            elif i == depth:
                in_channels = dimensions * (2 ** (i - 1))
                out_channels = dimensions * (2 ** (i - 1))
            else:
                in_channels = dimensions * (2 ** (i - 1))
                out_channels = 2 * in_channels
            pooling_layer = torch.nn.AvgPool2d(2)
            double_convolution_layer = double_convolution(
                                                          input_channels = in_channels,
                                                          mid_channels = in_channels,
                                                          output_channels = out_channels,
                                                          kernel_size = kernel_size,
                                                          bias = bias,
                                                          normalization = normalization,
                                                          activation = activation
                                                         )
            self.encoder.append(pooling_layer)
            self.encoder.append(double_convolution_layer)
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
                                in_channels = svf_in_channels,
                                out_channels = svf_mid_channels,
                                kernel_size = kernel_size,
                                padding = kernel_size // 2,
                                bias = bias
                               ),
                activation,
                torch.nn.Conv2d(
                                in_channels = svf_mid_channels,
                                out_channels = svf_mid_channels,
                                kernel_size = kernel_size,
                                padding = kernel_size // 2,
                                bias = bias
                               ),
                activation,
                torch.nn.Conv2d(
                                in_channels = svf_mid_channels,
                                out_channels = svf_out_channels,
                                kernel_size = kernel_size,
                                padding = kernel_size // 2,
                                bias = bias
                               ),
            )
            spatially_varying_kernel_generation.append(kernel_generation_block)
            self.spatially_varying_feature.append(spatially_varying_kernel_generation)
        self.decoder = torch.nn.ModuleList()
        global_feature_layer = global_feature_module(  # global feature layer
                                                     input_channels = dimensions * (2 ** (depth - 1)),
                                                     mid_channels = dimensions * (2 ** (depth - 1)),
                                                     output_channels = dimensions * (2 ** (depth - 1)),
                                                     kernel_size = kernel_size,
                                                     bias = bias,
                                                     activation = torch.nn.LeakyReLU(0.2, inplace = True)
                                                    )
        self.decoder.append(global_feature_layer)
        for i in range(depth, 0, -1):
            if i == 2:
                up_in_channels = (dimensions // 2) * (2 ** i)
                up_out_channels = up_in_channels
                up_mid_channels = up_in_channels
            elif i == 1:
                up_in_channels = dimensions * 2
                up_out_channels = dimensions
                up_mid_channels = up_out_channels
            else:
                up_in_channels = (dimensions // 2) * (2 ** i)
                up_out_channels = up_in_channels // 2
                up_mid_channels = up_in_channels
            upsample_layer = upsample_convtranspose2d_layer(
                                                            input_channels = up_in_channels,
                                                            output_channels = up_mid_channels,
                                                            kernel_size = 2,
                                                            stride = 2,
                                                            bias = bias,
                                                           )
            conv_layer = double_convolution(
                                            input_channels = up_mid_channels,
                                            output_channels = up_out_channels,
                                            kernel_size = kernel_size,
                                            bias = bias,
                                            normalization = normalization,
                                            activation = activation,
                                           )
            self.decoder.append(torch.nn.ModuleList([upsample_layer, conv_layer]))


    def forward(self, focal_surface, field):
        """
        Forward model.

        Parameters
        ----------
        focal_surface : torch.tensor
                        Input focal surface data.
                        Dimension: (1, 1, H, W)

        field         : torch.tensor
                        Input field data.
                        Dimension: (1, 6, H, W)

        Returns
        -------
        sv_kernel : list of torch.tensor
                    Learned spatially varying kernels.
                    Dimension of each element in the list: (1, C_i * kernel_size * kernel_size, H_i, W_i),
                    where C_i, H_i, and W_i represent the channel, height, and width
                    of each feature at a certain scale.
        """
        x = self.inc(torch.cat((focal_surface, field), dim = 1))
        downsampling_outputs = [focal_surface]
        downsampling_outputs.append(x)
        for i, down_layer in enumerate(self.encoder):
            x_down = down_layer(downsampling_outputs[-1])
            downsampling_outputs.append(x_down)
        sv_kernels = []
        for i, (up_layer, svf_layer) in enumerate(zip(self.decoder, self.spatially_varying_feature)):
            if i == 0:
                global_feature = up_layer(downsampling_outputs[-2], downsampling_outputs[-1])
                downsampling_outputs[-1] = global_feature
                sv_feature = [global_feature, downsampling_outputs[0]]
                for j in range(self.depth - i + 1):
                    sv_feature[1] = svf_layer[self.depth - i](sv_feature[1])
                    if j > 0:
                        sv_feature.append(svf_layer[j](downsampling_outputs[2 * j]))
                sv_feature = [sv_feature[0], sv_feature[1], sv_feature[4], sv_feature[2],
                              sv_feature[3]]
                sv_kernel = svf_layer[-1](torch.cat(sv_feature, dim = 1))
                sv_kernels.append(sv_kernel)
            else:
                x_up = up_layer[0](downsampling_outputs[-1],
                                   downsampling_outputs[2 * (self.depth + 1 - i) + 1])
                x_up = up_layer[1](x_up)
                downsampling_outputs[-1] = x_up
                sv_feature = [x_up, downsampling_outputs[0]]
                for j in range(self.depth - i + 1):
                    sv_feature[1] = svf_layer[self.depth - i](sv_feature[1])
                    if j > 0:
                        sv_feature.append(svf_layer[j](downsampling_outputs[2 * j]))
                if i == 1:
                    sv_feature = [sv_feature[0], sv_feature[1], sv_feature[3], sv_feature[2]]
                sv_kernel = svf_layer[-1](torch.cat(sv_feature, dim = 1))
                sv_kernels.append(sv_kernel)
        return sv_kernels


class spatially_adaptive_unet(torch.nn.Module):
    """
    Spatially varying U-Net model based on spatially adaptive convolution.

    References
    ----------

    Chuanjun Zheng, Yicheng Zhan, Liang Shi, Ozan Cakmakci, and Kaan Ak{\c{s}}it}. "Focal Surface Holographic Light Transport using Learned Spatially Adaptive Convolutions." SIGGRAPH Asia 2024 Technical Communications (SA Technical Communications '24),December,2024.
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
                 activation=torch.nn.LeakyReLU(0.2, inplace=True)
                ):
        """
        U-Net model.

        Parameters
        ----------
        depth          : int
                         Number of upsampling and downsampling layers.
        dimensions     : int
                         Number of dimensions.
        input_channels : int
                         Number of input channels.
        out_channels   : int
                         Number of output channels.
        bias           : bool
                         Set to True to let convolutional layers learn a bias term.
        normalization  : bool
                         If True, adds a Batch Normalization layer after the convolutional layer.
        activation     : torch.nn
                         Non-linear activation layer (e.g., torch.nn.ReLU(), torch.nn.Sigmoid()).
        """
        super().__init__()
        self.depth = depth
        self.out_channels = out_channels
        self.inc = convolution_layer(
                                     input_channels=input_channels,
                                     output_channels=dimensions,
                                     kernel_size=kernel_size,
                                     bias=bias,
                                     normalization=normalization,
                                     activation=activation
                                    )

        self.encoder = torch.nn.ModuleList()
        for i in range(self.depth + 1):  # Downsampling layers
            down_in_channels = dimensions * (2 ** i)
            down_out_channels = 2 * down_in_channels
            pooling_layer = torch.nn.AvgPool2d(2)
            double_convolution_layer = double_convolution(
                                                          input_channels=down_in_channels,
                                                          mid_channels=down_in_channels,
                                                          output_channels=down_in_channels,
                                                          kernel_size=kernel_size,
                                                          bias=bias,
                                                          normalization=normalization,
                                                          activation=activation
                                                         )
            sam = spatially_adaptive_module(
                                            input_channels=down_in_channels,
                                            output_channels=down_out_channels,
                                            kernel_size=kernel_size,
                                            bias=bias,
                                            activation=activation
                                           )
            self.encoder.append(torch.nn.ModuleList([pooling_layer, double_convolution_layer, sam]))
        self.global_feature_module = torch.nn.ModuleList()
        double_convolution_layer = double_convolution(
                                                      input_channels=dimensions * (2 ** (depth + 1)),
                                                      mid_channels=dimensions * (2 ** (depth + 1)),
                                                      output_channels=dimensions * (2 ** (depth + 1)),
                                                      kernel_size=kernel_size,
                                                      bias=bias,
                                                      normalization=normalization,
                                                      activation=activation
                                                     )
        global_feature_layer = global_feature_module(
                                                     input_channels=dimensions * (2 ** (depth + 1)),
                                                     mid_channels=dimensions * (2 ** (depth + 1)),
                                                     output_channels=dimensions * (2 ** (depth + 1)),
                                                     kernel_size=kernel_size,
                                                     bias=bias,
                                                     activation=torch.nn.LeakyReLU(0.2, inplace=True)
                                                    )
        self.global_feature_module.append(torch.nn.ModuleList([double_convolution_layer, global_feature_layer]))
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
                                     )
                )
                self.decoder.append(torch.nn.ModuleList([upsample_layer, conv_layer]))
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


    def forward(self, sv_kernel, field):
        """
        Forward model.

        Parameters
        ----------
        sv_kernel : list of torch.tensor
                    Learned spatially varying kernels.
                    Dimension of each element in the list: (1, C_i * kernel_size * kernel_size, H_i, W_i),
                    where C_i, H_i, and W_i represent the channel, height, and width
                    of each feature at a certain scale.

        field     : torch.tensor
                    Input field data.
                    Dimension: (1, 6, H, W)

        Returns
        -------
        target_field : torch.tensor
                       Estimated output.
                       Dimension: (1, 6, H, W)
        """
        x = self.inc(field)
        downsampling_outputs = [x]
        for i, down_layer in enumerate(self.encoder):
            x_down = down_layer[0](downsampling_outputs[-1])
            downsampling_outputs.append(x_down)
            sam_output = down_layer[2](x_down + down_layer[1](x_down), sv_kernel[self.depth - i])
            downsampling_outputs.append(sam_output)
        global_feature = self.global_feature_module[0][0](downsampling_outputs[-1])
        global_feature = self.global_feature_module[0][1](downsampling_outputs[-1], global_feature)
        downsampling_outputs.append(global_feature)
        x_up = downsampling_outputs[-1]
        for i, up_layer in enumerate(self.decoder):
            x_up = up_layer[0](x_up, downsampling_outputs[2 * (self.depth - i)])
            x_up = up_layer[1](x_up)
        result = x_up
        return result
