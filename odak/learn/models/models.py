import torch
from .components import double_convolution, downsample_layer, upsample_layer


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
                            Model type: `conventional`, `SIREN`, `FILM SIREN`.
                            `conventional` refers to a standard multi layer perceptron.
                            For `SIREN,` see: Sitzmann, Vincent, et al. "Implicit neural representations with periodic activation functions." Advances in neural information processing systems 33 (2020): 7462-7473.
                            For `FILM SIREN,` see: Chan, Eric R., et al. "pi-gan: Periodic implicit generative adversarial networks for 3d-aware image synthesis." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.
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
                result = self.activation(self.siren_multiplier * result)
            elif self.model_type == 'SIREN':
                result = torch.sin(result)
            elif self.model_type == 'FILM SIREN':
                result = torch.sin(self.alpha[layer_id][0] * result + self.alpha[layer_id][1])
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
        factor = 2 if bilinear else 1        
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

