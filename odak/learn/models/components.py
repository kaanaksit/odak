import torch
import math


def gaussian(x, multiplier = 1.):
    """
    A Gaussian non-linear activation.
    For more details: Ramasinghe, Sameera, and Simon Lucey. "Beyond periodicity: Towards a unifying framework for activations in coordinate-mlps." In European Conference on Computer Vision, pp. 142-158. Cham: Springer Nature Switzerland, 2022.

    Parameters
    ----------
    x            : float or torch.tensor
                   Input data.
    multiplier   : float or torch.tensor
                   Multiplier.

    Returns
    -------
    result       : float or torch.tensor
                   Ouput data.
    """
    result = torch.exp(- (multiplier * x) ** 2)
    return result


def swish(x):
    """
    A swish non-linear activation.
    For more details: https://en.wikipedia.org/wiki/Swish_function

    Parameters
    -----------
    x              : float or torch.tensor
                     Input.

    Returns
    -------
    out            : float or torch.tensor
                     Output.
    """
    out = x * torch.sigmoid(x)
    return out


class residual_layer(torch.nn.Module):
    """
    A residual layer.
    """
    def __init__(
                 self,
                 input_channels = 2,
                 mid_channels = 16,
                 kernel_size = 3,
                 bias = False,
                 activation = torch.nn.ReLU()
                ):
        """
        A convolutional layer class.


        Parameters
        ----------
        input_channels  : int
                          Number of input channels.
        mid_channels    : int
                          Number of middle channels.
        kernel_size     : int
                          Kernel size.
        bias            : bool 
                          Set to True to let convolutional layers have bias term.
        activation      : torch.nn
                          Nonlinear activation layer to be used. If None, uses torch.nn.ReLU().
        """
        super().__init__()
        self.activation = activation
        self.convolution = double_convolution(
                                              input_channels,
                                              mid_channels = mid_channels,
                                              output_channels = input_channels,
                                              kernel_size = kernel_size,
                                              bias = bias,
                                              activation = activation
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
        x0 = self.convolution(x)
        return x + x0


class convolution_layer(torch.nn.Module):
    """
    A convolution layer.
    """
    def __init__(
                 self,
                 input_channels = 2,
                 output_channels = 2,
                 kernel_size = 3,
                 bias = False,
                 stride = 1,
                 normalization = True,
                 activation = torch.nn.ReLU()
                ):
        """
        A convolutional layer class.


        Parameters
        ----------
        input_channels  : int
                          Number of input channels.
        output_channels : int
                          Number of output channels.
        kernel_size     : int
                          Kernel size.
        bias            : bool
                          Set to True to let convolutional layers have bias term.
        normalization   : bool
                          If True, adds a Batch Normalization layer after the convolutional layer.
        activation      : torch.nn
                          Nonlinear activation layer to be used. If None, uses torch.nn.ReLU().
        """
        super().__init__()
        layers = [
            torch.nn.Conv2d(
                            input_channels,
                            output_channels,
                            kernel_size = kernel_size,
                            stride = stride,
                            padding = kernel_size // 2,
                            bias = bias
                           )
        ]
        if normalization:
            layers.append(torch.nn.BatchNorm2d(output_channels))
        if activation:
            layers.append(activation)
        self.model = torch.nn.Sequential(*layers)


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
        result = self.model(x)
        return result


class double_convolution(torch.nn.Module):
    """
    A double convolution layer.
    """
    def __init__(
                 self,
                 input_channels = 2,
                 mid_channels = None,
                 output_channels = 2,
                 kernel_size = 3, 
                 bias = False,
                 normalization = True,
                 activation = torch.nn.ReLU()
                ):
        """
        Double convolution model.


        Parameters
        ----------
        input_channels  : int
                          Number of input channels.
        mid_channels    : int
                          Number of channels in the hidden layer between two convolutions.
        output_channels : int
                          Number of output channels.
        kernel_size     : int
                          Kernel size.
        bias            : bool 
                          Set to True to let convolutional layers have bias term.
        normalization   : bool
                          If True, adds a Batch Normalization layer after the convolutional layer.
        activation      : torch.nn
                          Nonlinear activation layer to be used. If None, uses torch.nn.ReLU().
        """
        super().__init__()
        if isinstance(mid_channels, type(None)):
            mid_channels = output_channels
        self.activation = activation
        self.model = torch.nn.Sequential(
                                         convolution_layer(
                                                           input_channels = input_channels,
                                                           output_channels = mid_channels,
                                                           kernel_size = kernel_size,
                                                           bias = bias,
                                                           normalization = normalization,
                                                           activation = self.activation
                                                          ),
                                         convolution_layer(
                                                           input_channels = mid_channels,
                                                           output_channels = output_channels,
                                                           kernel_size = kernel_size,
                                                           bias = bias,
                                                           normalization = normalization,
                                                           activation = self.activation
                                                          )
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
        result = self.model(x)
        return result


class normalization(torch.nn.Module):
    """
    A normalization layer.
    """
    def __init__(
                 self,
                 dim = 1,
                ):
        """
        Normalization layer.


        Parameters
        ----------
        dim             : int
                          Dimension (axis) to normalize.
        """
        super().__init__()
        self.k = torch.nn.Parameter(torch.ones(1, dim, 1, 1))


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
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        result =  (x - mean) * (var + eps).rsqrt() * self.k
        return result 

 
class residual_attention_layer(torch.nn.Module):
    """
    A residual block with an attention layer.
    """
    def __init__(
                 self,
                 input_channels = 2,
                 output_channels = 2,
                 kernel_size = 1,
                 bias = False,
                 activation = torch.nn.ReLU()
                ):
        """
        An attention layer class.


        Parameters
        ----------
        input_channels  : int or optioal
                          Number of input channels.
        output_channels : int or optional
                          Number of middle channels.
        kernel_size     : int or optional
                          Kernel size.
        bias            : bool or optional
                          Set to True to let convolutional layers have bias term.
        activation      : torch.nn or optional
                          Nonlinear activation layer to be used. If None, uses torch.nn.ReLU().
        """
        super().__init__()
        self.activation = activation
        self.convolution0 = torch.nn.Sequential(
                                                torch.nn.Conv2d(
                                                                input_channels,
                                                                output_channels,
                                                                kernel_size = kernel_size,
                                                                padding = kernel_size // 2,
                                                                bias = bias
                                                               ),
                                                torch.nn.BatchNorm2d(output_channels)
                                               )
        self.convolution1 = torch.nn.Sequential(
                                                torch.nn.Conv2d(
                                                                input_channels,
                                                                output_channels,
                                                                kernel_size = kernel_size,
                                                                padding = kernel_size // 2,
                                                                bias = bias
                                                               ),
                                                torch.nn.BatchNorm2d(output_channels)
                                               )
        self.final_layer = torch.nn.Sequential(
                                               self.activation,
                                               torch.nn.Conv2d(
                                                               output_channels,
                                                               output_channels,
                                                               kernel_size = kernel_size,
                                                               padding = kernel_size // 2,
                                                               bias = bias
                                                              )
                                              )


    def forward(self, x0, x1):
        """
        Forward model.
        
        Parameters
        ----------
        x0             : torch.tensor
                         First input data.
                    
        x1             : torch.tensor
                         Seconnd input data.
      
 
        Returns
        ----------
        result        : torch.tensor
                        Estimated output.      
        """
        y0 = self.convolution0(x0)
        y1 = self.convolution1(x1)
        y2 = torch.add(y0, y1)
        result = self.final_layer(y2) * x0
        return result

 
class non_local_layer(torch.nn.Module):
    """
    Self-Attention Layer [zi = Wzyi + xi] (non-local block : ref https://arxiv.org/abs/1711.07971)
    """
    def __init__(
                 self,
                 input_channels = 1024,
                 bottleneck_channels = 512,
                 kernel_size = 1,
                 bias = False,
                ):
        """

        Parameters
        ----------
        input_channels      : int
                              Number of input channels.
        bottleneck_channels : int
                              Number of middle channels.
        kernel_size         : int
                              Kernel size.
        bias                : bool 
                              Set to True to let convolutional layers have bias term.
        """
        super(non_local_layer, self).__init__()
        self.input_channels = input_channels
        self.bottleneck_channels = bottleneck_channels
        self.g = torch.nn.Conv2d(
                                 self.input_channels, 
                                 self.bottleneck_channels,
                                 kernel_size = kernel_size,
                                 padding = kernel_size // 2,
                                 bias = bias
                                )
        self.W_z = torch.nn.Sequential(
                                       torch.nn.Conv2d(
                                                       self.bottleneck_channels,
                                                       self.input_channels, 
                                                       kernel_size = kernel_size,
                                                       bias = bias,
                                                       padding = kernel_size // 2
                                                      ),
                                       torch.nn.BatchNorm2d(self.input_channels)
                                      )
        torch.nn.init.constant_(self.W_z[1].weight, 0)   
        torch.nn.init.constant_(self.W_z[1].bias, 0)


    def forward(self, x):
        """
        Forward model [zi = Wzyi + xi]
        
        Parameters
        ----------
        x               : torch.tensor
                          First input data.                       
    

        Returns
        ----------
        z               : torch.tensor
                          Estimated output.
        """
        batch_size, channels, height, width = x.size()
        theta = x.view(batch_size, channels, -1).permute(0, 2, 1)
        phi = x.view(batch_size, channels, -1).permute(0, 2, 1)
        g = self.g(x).view(batch_size, self.bottleneck_channels, -1).permute(0, 2, 1)
        attn = torch.bmm(theta, phi.transpose(1, 2)) / (height * width)
        attn = torch.nn.functional.softmax(attn, dim=-1)
        y = torch.bmm(attn, g).permute(0, 2, 1).contiguous().view(batch_size, self.bottleneck_channels, height, width)
        W_y = self.W_z(y)
        z = W_y + x
        return z


class downsample_layer(torch.nn.Module):
    """
    A downscaling component followed by a double convolution.
    """
    def __init__(
                 self,
                 input_channels,
                 output_channels,
                 kernel_size = 3,
                 bias = False,
                 activation = torch.nn.ReLU()
                ):
        """
        A downscaling component with a double convolution.

        Parameters
        ----------
        input_channels  : int
                          Number of input channels.
        output_channels : int
                          Number of output channels.
        kernel_size     : int
                          Kernel size.
        bias            : bool 
                          Set to True to let convolutional layers have bias term.
        activation      : torch.nn
                          Nonlinear activation layer to be used. If None, uses torch.nn.ReLU().
        """
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
                                                torch.nn.MaxPool2d(2),
                                                double_convolution(
                                                                   input_channels = input_channels,
                                                                   mid_channels = output_channels,
                                                                   output_channels = output_channels,
                                                                   kernel_size = kernel_size,
                                                                   bias = bias,
                                                                   activation = activation
                                                                  )
                                               )


    def forward(self, x):
        """
        Forward model.
        
        Parameters
        ----------
        x              : torch.tensor
                         First input data.
                    
      
 
        Returns
        ----------
        result        : torch.tensor
                        Estimated output.      
        """
        result = self.maxpool_conv(x)
        return result


class upsample_layer(torch.nn.Module):
    """
    An upsampling convolutional layer.
    """
    def __init__(
                 self,
                 input_channels,
                 output_channels,
                 kernel_size = 3,
                 bias = False,
                 activation = torch.nn.ReLU(),
                 bilinear = True
                ):
        """
        A downscaling component with a double convolution.

        Parameters
        ----------
        input_channels  : int
                          Number of input channels.
        output_channels : int
                          Number of output channels.
        kernel_size     : int
                          Kernel size.
        bias            : bool 
                          Set to True to let convolutional layers have bias term.
        activation      : torch.nn
                          Nonlinear activation layer to be used. If None, uses torch.nn.ReLU().
        bilinear        : bool
                          If set to True, bilinear sampling is used.
        """
        super(upsample_layer, self).__init__()
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
            self.conv = double_convolution(
                                           input_channels = input_channels + output_channels,
                                           mid_channels = input_channels // 2,
                                           output_channels = output_channels,
                                           kernel_size = kernel_size,
                                           bias = bias,
                                           activation = activation
                                          )
        else:
            self.up = torch.nn.ConvTranspose2d(input_channels , input_channels // 2, kernel_size = 2, stride = 2)
            self.conv = double_convolution(
                                           input_channels = input_channels,
                                           mid_channels = output_channels,
                                           output_channels = output_channels,
                                           kernel_size = kernel_size,
                                           bias = bias,
                                           activation = activation
                                          )


    def forward(self, x1, x2):
        """
        Forward model.
        
        Parameters
        ----------
        x1             : torch.tensor
                         First input data.
        x2             : torch.tensor
                         Second input data.
 

        Returns
        ----------
        result        : torch.tensor
                        Result of the forward operation
        """ 
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim = 1)
        result = self.conv(x)
        return result


class channel_gate(torch.nn.Module):
    """
    Channel attention module with various pooling strategies.
    This class is heavily inspired https://github.com/Jongchan/attention-module/commit/e4ee180f1335c09db14d39a65d97c8ca3d1f7b16 (MIT License).
    """
    def __init__(
                 self, 
                 gate_channels, 
                 reduction_ratio = 16, 
                 pool_types = ['avg', 'max']
                ):
        """
        Initializes the channel gate module.

        Parameters
        ----------
        gate_channels   : int
                          Number of channels of the input feature map.
        reduction_ratio : int
                          Reduction ratio for the intermediate layer.
        pool_types      : list
                          List of pooling operations to apply.
        """
        super().__init__()
        self.gate_channels = gate_channels
        hidden_channels = gate_channels // reduction_ratio
        if hidden_channels == 0:
            hidden_channels = 1
        self.mlp = torch.nn.Sequential(
                                       convolutional_block_attention.Flatten(),
                                       torch.nn.Linear(gate_channels, hidden_channels),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(hidden_channels, gate_channels)
                                      )
        self.pool_types = pool_types


    def forward(self, x):
        """
        Forward pass of the ChannelGate module.

        Applies channel-wise attention to the input tensor.

        Parameters
        ----------
        x            : torch.tensor
                       Input tensor to the ChannelGate module.

        Returns
        -------
        output       : torch.tensor
                       Output tensor after applying channel attention.
        """
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool = torch.nn.functional.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            elif pool_type == 'max':
                pool = torch.nn.functional.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            channel_att_raw = self.mlp(pool)
            channel_att_sum = channel_att_raw if channel_att_sum is None else channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        output = x * scale
        return output


class spatial_gate(torch.nn.Module):
    """
    Spatial attention module that applies a convolution layer after channel pooling.
    This class is heavily inspired by https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py.
    """
    def __init__(self):
        """
        Initializes the spatial gate module.
        """
        super().__init__()
        kernel_size = 7
        self.spatial = convolution_layer(2, 1, kernel_size, bias = False, activation = torch.nn.Identity())


    def channel_pool(self, x):
        """
        Applies max and average pooling on the channels.

        Parameters
        ----------
        x             : torch.tensor
                        Input tensor.

        Returns
        -------
        output        : torch.tensor
                        Output tensor.
        """
        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        avg_pool = torch.mean(x, 1).unsqueeze(1)
        output = torch.cat((max_pool, avg_pool), dim=1)
        return output


    def forward(self, x):
        """
        Forward pass of the SpatialGate module.

        Applies spatial attention to the input tensor.

        Parameters
        ----------
        x            : torch.tensor
                       Input tensor to the SpatialGate module.

        Returns
        -------
        scaled_x     : torch.tensor
                       Output tensor after applying spatial attention.
        """
        x_compress = self.channel_pool(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        scaled_x = x * scale
        return scaled_x
    
    
class convolutional_block_attention(torch.nn.Module):
    """
    Convolutional Block Attention Module (CBAM) class. 
    This class is heavily inspired https://github.com/Jongchan/attention-module/commit/e4ee180f1335c09db14d39a65d97c8ca3d1f7b16 (MIT License).
    """
    def __init__(
                 self, 
                 gate_channels, 
                 reduction_ratio = 16, 
                 pool_types = ['avg', 'max'], 
                 no_spatial = False
                ):
        """
        Initializes the convolutional block attention module.

        Parameters
        ----------
        gate_channels   : int
                          Number of channels of the input feature map.
        reduction_ratio : int
                          Reduction ratio for the channel attention.
        pool_types      : list
                          List of pooling operations to apply for channel attention.
        no_spatial      : bool
                          If True, spatial attention is not applied.
        """
        super(convolutional_block_attention, self).__init__()
        self.channel_gate = channel_gate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.spatial_gate = spatial_gate()

    
    class Flatten(torch.nn.Module):
        """
        Flattens the input tensor to a 2D matrix.
        """
        def forward(self, x):
            return x.view(x.size(0), -1)


    def forward(self, x):
        """
        Forward pass of the convolutional block attention module.

        Parameters
        ----------
        x            : torch.tensor
                       Input tensor to the CBAM module.

        Returns
        -------
        x_out        : torch.tensor
                       Output tensor after applying channel and spatial attention.
        """
        x_out = self.channel_gate(x)
        if not self.no_spatial:
            x_out = self.spatial_gate(x_out)
        return x_out


class positional_encoder(torch.nn.Module):
    """
    A positional encoder module.
    """
    
    def __init__(self, L):
        """
        A positional encoder module.

        Parameters
        ----------
        L                   : int
                              Positional encoding level.
        """
        super(positional_encoder, self).__init__()
        self.L = L


    def forward(self, x):
        """
        Forward model.
        
        Parameters
        ----------
        x               : torch.tensor
                          Input data.

        Returns
        ----------
        result          : torch.tensor
                          Result of the forward operation
        """
        B, C = x.shape
        x = x.view(B, C, 1)
        results = [x]
        for i in range(1, self.L + 1):
            freq = (2 ** i) * math.pi
            cos_x = torch.cos(freq * x)
            sin_x = torch.sin(freq * x)
            results.append(cos_x)
            results.append(sin_x)
        results = torch.cat(results, dim=2)
        results = results.permute(0, 2, 1)
        results = results.reshape(B, -1)
        return results


class upsample_convtranspose2d_layer(torch.nn.Module):
    """
    An upsampling convtranspose2d layer.
    """
    def __init__(
                 self,
                 input_channels,
                 output_channels,
                 kernel_size = 2,
                 stride = 2,
                 bias = False,
                ):
        """
        A downscaling component with a double convolution.

        Parameters
        ----------
        input_channels  : int
                          Number of input channels.
        output_channels : int
                          Number of output channels.
        kernel_size     : int
                          Kernel size.
        bias            : bool
                          Set to True to let convolutional layers have bias term.
        """
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(
                                           in_channels = input_channels,
                                           out_channels = output_channels,
                                           bias = bias,
                                           kernel_size = kernel_size,
                                           stride = stride
                                          )

    def forward(self, x1, x2):
        """
        Forward model.

        Parameters
        ----------
        x1             : torch.tensor
                         First input data.
        x2             : torch.tensor
                         Second input data.


        Returns
        ----------
        result        : torch.tensor
                        Result of the forward operation
        """
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
        result = x1 + x2
        return result


class global_transformations(torch.nn.Module):
    """
    A global feature layer that processes global features from input channels and
    applies learned transformations to another input tensor.

    This implementation is adapted from RSGUnet:
    https://github.com/MTLab/rsgunet_image_enhance.

    Reference:
    J. Huang, P. Zhu, M. Geng et al. "Range Scaling Global U-Net for Perceptual Image Enhancement on Mobile Devices."
    """
    def __init__(
                 self,
                 input_channels,
                 output_channels
                ):
        """
        A global feature layer.

        Parameters
        ----------
        input_channels  : int
                          Number of input channels.
        output_channels : int
                          Number of output channels.
        """
        super().__init__()
        self.global_feature_1 = torch.nn.Sequential(
            torch.nn.Linear(input_channels, output_channels),
            torch.nn.LeakyReLU(0.2, inplace = True),
        )
        self.global_feature_2 = torch.nn.Sequential(
            torch.nn.Linear(output_channels, output_channels),
            torch.nn.LeakyReLU(0.2, inplace = True)
        )


    def forward(self, x1, x2):
        """
        Forward model.

        Parameters
        ----------
        x1             : torch.tensor
                         First input data.
        x2             : torch.tensor
                         Second input data.

        Returns
        ----------
        result        : torch.tensor
                        Estimated output.
        """
        y = torch.mean(x2, dim = (2, 3))
        y1 = self.global_feature_1(y)
        y2 = self.global_feature_2(y1)
        y1 = y1.unsqueeze(2).unsqueeze(3)
        y2 = y2.unsqueeze(2).unsqueeze(3)
        result = x1 * y1 + y2
        return result


class global_feature_module(torch.nn.Module):
    """
    A global feature layer that processes global features from input channels and
    applies them to another input tensor via learned transformations.
    """
    def __init__(
                 self,
                 input_channels,
                 mid_channels,
                 output_channels,
                 kernel_size,
                 bias = False,
                 normalization = False,
                 activation = torch.nn.ReLU()
                ):
        """
        A global feature layer.

        Parameters
        ----------
        input_channels  : int
                          Number of input channels.
        mid_channels  : int
                          Number of mid channels.
        output_channels : int
                          Number of output channels.
        kernel_size     : int
                          Kernel size.
        bias            : bool
                          Set to True to let convolutional layers have bias term.
        normalization   : bool
                          If True, adds a Batch Normalization layer after the convolutional layer.
        activation      : torch.nn
                          Nonlinear activation layer to be used. If None, uses torch.nn.ReLU().
        """
        super().__init__()
        self.transformations_1 = global_transformations(input_channels, output_channels)
        self.global_features_1 = double_convolution(
                                                    input_channels = input_channels,
                                                    mid_channels = mid_channels,
                                                    output_channels = output_channels,
                                                    kernel_size = kernel_size,
                                                    bias = bias,
                                                    normalization = normalization,
                                                    activation = activation
                                                   )
        self.global_features_2 = double_convolution(
                                                    input_channels = input_channels,
                                                    mid_channels = mid_channels,
                                                    output_channels = output_channels,
                                                    kernel_size = kernel_size,
                                                    bias = bias,
                                                    normalization = normalization,
                                                    activation = activation
                                                   )
        self.transformations_2 = global_transformations(input_channels, output_channels)


    def forward(self, x1, x2):
        """
        Forward model.

        Parameters
        ----------
        x1             : torch.tensor
                         First input data.
        x2             : torch.tensor
                         Second input data.

        Returns
        ----------
        result        : torch.tensor
                        Estimated output.
        """
        global_tensor_1 = self.transformations_1(x1, x2)
        y1 = self.global_features_1(global_tensor_1)
        y2 = self.global_features_2(y1)
        global_tensor_2 = self.transformations_2(y1, y2)
        return global_tensor_2


class spatially_adaptive_convolution(torch.nn.Module):
    """
    A spatially adaptive convolution layer.

    References
    ----------

    C. Zheng et al. "Focal Surface Holographic Light Transport using Learned Spatially Adaptive Convolutions."
    C, Xu et al. "Squeezesegv3: Spatially-adaptive Convolution for Efficient Point-Cloud Segmentation."
    C. Zheng et al. "Windowing Decomposition Convolutional Neural Network for Image Enhancement."
    """
    def __init__(
                 self,
                 input_channels = 2,
                 output_channels = 2,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 bias = False,
                 activation = torch.nn.LeakyReLU(0.2, inplace = True)
                ):
        """
        Initializes a spatially adaptive convolution layer.

        Parameters
        ----------
        input_channels  : int
                          Number of input channels.
        output_channels : int
                          Number of output channels.
        kernel_size     : int
                          Size of the convolution kernel.
        stride          : int
                          Stride of the convolution.
        padding         : int
                          Padding added to both sides of the input.
        bias            : bool
                          If True, includes a bias term in the convolution.
        activation      : torch.nn.Module
                          Activation function to apply. If None, no activation is applied.
        """
        super(spatially_adaptive_convolution, self).__init__()
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.padding = padding
        self.standard_convolution = torch.nn.Conv2d(
                                                    in_channels = input_channels,
                                                    out_channels = self.output_channels,
                                                    kernel_size = kernel_size,
                                                    stride = stride,
                                                    padding = padding,
                                                    bias = bias
                                                   )
        self.weight = torch.nn.Parameter(data = self.standard_convolution.weight, requires_grad = True)
        self.activation = activation


    def forward(self, x, sv_kernel_feature):
        """
        Forward pass for the spatially adaptive convolution layer.

        Parameters
        ----------
        x                  : torch.tensor
                            Input data tensor.
                            Dimension: (1, C, H, W)
        sv_kernel_feature   : torch.tensor
                            Spatially varying kernel features.
                            Dimension: (1, C_i * kernel_size * kernel_size, H, W)

        Returns
        -------
        sa_output          : torch.tensor
                            Estimated output tensor.
                            Dimension: (1, output_channels, H_out, W_out)
        """
        # Pad input and sv_kernel_feature if necessary
        if sv_kernel_feature.size(-1) * self.stride != x.size(-1) or sv_kernel_feature.size(
                -2) * self.stride != x.size(-2):
            diffY = sv_kernel_feature.size(-2) % self.stride
            diffX = sv_kernel_feature.size(-1) % self.stride
            sv_kernel_feature = torch.nn.functional.pad(sv_kernel_feature, (diffX // 2, diffX - diffX // 2,
                                                                            diffY // 2, diffY - diffY // 2))
            diffY = x.size(-2) % self.stride
            diffX = x.size(-1) % self.stride
            x = torch.nn.functional.pad(x, (diffX // 2, diffX - diffX // 2,
                                            diffY // 2, diffY - diffY // 2))

        # Unfold the input tensor for matrix multiplication
        input_feature = torch.nn.functional.unfold(
                                                   x,
                                                   kernel_size = (self.kernel_size, self.kernel_size),
                                                   stride = self.stride,
                                                   padding = self.padding
                                                  )

        # Resize sv_kernel_feature to match the input feature
        sv_kernel = sv_kernel_feature.reshape(
                                              1,
                                              self.input_channels * self.kernel_size * self.kernel_size,
                                              (x.size(-2) // self.stride) * (x.size(-1) // self.stride)
                                             )

        # Resize weight to match the input channels and kernel size
        si_kernel = self.weight.reshape(
                                        self.weight_output_channels,
                                        self.input_channels * self.kernel_size * self.kernel_size
                                       )

        # Apply spatially varying kernels
        sv_feature = input_feature * sv_kernel

        # Perform matrix multiplication
        sa_output = torch.matmul(si_kernel, sv_feature).reshape(
                                                                1, self.weight_output_channels,
                                                                (x.size(-2) // self.stride),
                                                                (x.size(-1) // self.stride)
                                                               )
        return sa_output


class spatially_adaptive_module(torch.nn.Module):
    """
    A spatially adaptive module that combines learned spatially adaptive convolutions.

    References
    ----------

    Chuanjun Zheng, Yicheng Zhan, Liang Shi, Ozan Cakmakci, and Kaan Ak{\c{s}}it}. "Focal Surface Holographic Light Transport using Learned Spatially Adaptive Convolutions." SIGGRAPH Asia 2024 Technical Communications (SA Technical Communications '24),December,2024.
    """
    def __init__(
                 self,
                 input_channels = 2,
                 output_channels = 2,
                 kernel_size = 3,
                 stride = 1,
                 padding = 1,
                 bias = False,
                 activation = torch.nn.LeakyReLU(0.2, inplace = True)
                ):
        """
        Initializes a spatially adaptive module.

        Parameters
        ----------
        input_channels  : int
                          Number of input channels.
        output_channels : int
                          Number of output channels.
        kernel_size     : int
                          Size of the convolution kernel.
        stride          : int
                          Stride of the convolution.
        padding         : int
                          Padding added to both sides of the input.
        bias            : bool
                          If True, includes a bias term in the convolution.
        activation      : torch.nn
                          Nonlinear activation layer to be used. If None, uses torch.nn.ReLU().
        """
        super(spatially_adaptive_module, self).__init__()
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.padding = padding
        self.weight_output_channels = self.output_channels - 1
        self.standard_convolution = torch.nn.Conv2d(
                                                    in_channels = input_channels,
                                                    out_channels = self.weight_output_channels,
                                                    kernel_size = kernel_size,
                                                    stride = stride,
                                                    padding = padding,
                                                    bias = bias
                                                   )
        self.weight = torch.nn.Parameter(data = self.standard_convolution.weight, requires_grad = True)
        self.activation = activation


    def forward(self, x, sv_kernel_feature):
        """
        Forward pass for the spatially adaptive module.

        Parameters
        ----------
        x                  : torch.tensor
                            Input data tensor.
                            Dimension: (1, C, H, W)
        sv_kernel_feature   : torch.tensor
                            Spatially varying kernel features.
                            Dimension: (1, C_i * kernel_size * kernel_size, H, W)

        Returns
        -------
        output             : torch.tensor
                            Combined output tensor from standard and spatially adaptive convolutions.
                            Dimension: (1, output_channels, H_out, W_out)
        """
        # Pad input and sv_kernel_feature if necessary
        if sv_kernel_feature.size(-1) * self.stride != x.size(-1) or sv_kernel_feature.size(
                -2) * self.stride != x.size(-2):
            diffY = sv_kernel_feature.size(-2) % self.stride
            diffX = sv_kernel_feature.size(-1) % self.stride
            sv_kernel_feature = torch.nn.functional.pad(sv_kernel_feature, (diffX // 2, diffX - diffX // 2,
                                                                            diffY // 2, diffY - diffY // 2))
            diffY = x.size(-2) % self.stride
            diffX = x.size(-1) % self.stride
            x = torch.nn.functional.pad(x, (diffX // 2, diffX - diffX // 2,
                                            diffY // 2, diffY - diffY // 2))

        # Unfold the input tensor for matrix multiplication
        input_feature = torch.nn.functional.unfold(
                                                   x,
                                                   kernel_size = (self.kernel_size, self.kernel_size),
                                                   stride = self.stride,
                                                   padding = self.padding
                                                  )

        # Resize sv_kernel_feature to match the input feature
        sv_kernel = sv_kernel_feature.reshape(
                                              1,
                                              self.input_channels * self.kernel_size * self.kernel_size,
                                              (x.size(-2) // self.stride) * (x.size(-1) // self.stride)
                                             )

        # Apply sv_kernel to the input_feature
        sv_feature = input_feature * sv_kernel

        # Original spatially varying convolution output
        sv_output = torch.sum(sv_feature, dim = 1).reshape(
                                                           1,
                                                            1,
                                                            (x.size(-2) // self.stride),
                                                            (x.size(-1) // self.stride)
                                                           )

        # Reshape weight for spatially adaptive convolution
        si_kernel = self.weight.reshape(
                                        self.weight_output_channels,
                                        self.input_channels * self.kernel_size * self.kernel_size
                                       )

        # Apply si_kernel on sv convolution output
        sa_output = torch.matmul(si_kernel, sv_feature).reshape(
                                                                1, self.weight_output_channels,
                                                                (x.size(-2) // self.stride),
                                                                (x.size(-1) // self.stride)
                                                               )

        # Combine the outputs and apply activation function
        output = self.activation(torch.cat((sv_output, sa_output), dim = 1))
        return output
