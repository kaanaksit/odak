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
                 mid_channels = 8,
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
    An attention layer.
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
        self.convolution = torch.nn.Sequential(
                                         torch.nn.Conv2d(
                                                         input_channels,
                                                         output_channels,
                                                         kernel_size = kernel_size,
                                                         padding = kernel_size // 2,
                                                         bias = bias
                                                        ),
                                         torch.nn.BatchNorm2d(output_channels),
                                        )


    def forward(self, x_1, x_2):
        """
        Forward model.
        
        Parameters
        ----------
        x_1            : torch.tensor
                         First input data.
                    
        x_2            : torch.tensor
                         Seconnd input data.
      
 
        Returns
        ----------
        result        : torch.tensor
                        Estimated output.      
        """
        x_1_out = self.convolution(x_1)
        x_2_out = self.convolution(x_2)
        result = self.activation(x_1_out + x_2_out) * x_1
        return result

