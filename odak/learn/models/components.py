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
                 bias = False
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
        """
        super().__init__()
        self.convolution0 = convolution_layer(
                                              input_channels,
                                              mid_channels,
                                              kernel_size = kernel_size,
                                              bias = bias
                                             )
        self.convolution1 = torch.nn.Conv2d(
                                            mid_channels,
                                            input_channels,
                                            kernel_size = kernel_size,
                                            padding = kernel_size // 2,
                                            bias = bias
                                           )
        self.output_layer = torch.nn.Sequential(
                                                torch.nn.BatchNorm2d(input_channels),
                                                torch.nn.Tanh()
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
        x0 = self.convolution0(x)
        x1 = self.convolution1(x0)
        x2 = x + x1
        result = self.output_layer(x2)
        return result




class convolution_layer(torch.nn.Module):
    """
    A convolution layer.
    """
    def __init__(
                 self,
                 input_channels = 2,
                 output_channels = 2,
                 kernel_size = 3,
                 bias = False
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
        """
        super().__init__()
        self.model = torch.nn.Sequential(
                                         torch.nn.Conv2d(
                                                         input_channels,
                                                         output_channels,
                                                         kernel_size = kernel_size,
                                                         padding = kernel_size // 2,
                                                         bias = bias
                                                        ),
                                         torch.nn.BatchNorm2d(output_channels),
                                         torch.nn.Tanh()
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
                 bias = False
                ):
        """
        Double convolution model.


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
        self.model = torch.nn.Sequential(
                                         convolution_layer(
                                                           input_channels = input_channels,
                                                           output_channels = mid_channels,
                                                           kernel_size = kernel_size,
                                                           bias = bias
                                                          ),
                                         convolution_layer(
                                                           input_channels = mid_channels,
                                                           output_channels = output_channels,
                                                           kernel_size = kernel_size,
                                                           bias = bias
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
