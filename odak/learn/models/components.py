import torch


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
        activation      : torch.nn
                          Nonlinear activation layer to be used. If None, uses torch.nn.ReLU().
        """
        super().__init__()
        self.activation = activation
        self.model = torch.nn.Sequential(
                                         torch.nn.Conv2d(
                                                         input_channels,
                                                         output_channels,
                                                         kernel_size = kernel_size,
                                                         padding = kernel_size // 2,
                                                         bias = bias
                                                        ),
                                         torch.nn.BatchNorm2d(output_channels),
                                         self.activation
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
                                                           activation = self.activation
                                                          ),
                                         convolution_layer(
                                                           input_channels = mid_channels,
                                                           output_channels = output_channels,
                                                           kernel_size = kernel_size,
                                                           bias = bias,
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
                                           input_channels = input_channels,
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
        x              : torch.tensor
                         First input data.
                    
      
 
        Returns
        ----------
        result        : torch.tensor
        """ 
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim = 1)
        result = self.conv(x)
        return result
