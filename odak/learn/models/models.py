import torch
from .components import double_convolution, downsample_layer, upsample_layer


class unet(torch.nn.Module):
    """
    A U-Net model, heavily inspired from `https://github.com/milesial/Pytorch-UNet/tree/master/unet` and more can be read from Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015.
    """
    def __init__(
                 self, 
                 dimensions = 64, 
                 input_channels = 2, 
                 output_channels = 1, 
                 bilinear = True,
                 kernel_size = 3,
                 bias = True,
                 activation = torch.nn.ReLU()
                ):
        """
        U-Net model.

        Parameters
        ----------
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
        self.down1 = downsample_layer(
                                      dimensions, 
                                      dimensions * 2,
                                      kernel_size = kernel_size,
                                      bias = bias,
                                      activation = activation
                                     )
        self.down2 = downsample_layer(
                                      dimensions * 2, 
                                      dimensions * 4,
                                      kernel_size = kernel_size,
                                      bias = bias,
                                      activation = activation
                                     )
        self.down3 = downsample_layer(
                                      dimensions * 4, 
                                      dimensions * 8,
                                      kernel_size = kernel_size,
                                      bias = bias,
                                      activation = activation
                                     )
        factor = 2 if bilinear else 1
        self.down4 = downsample_layer(
                                      dimensions * 8, 
                                      dimensions * 16 // factor,
                                      kernel_size = kernel_size,
                                      bias = bias,
                                      activation = activation
                                     )
        self.up1 = upsample_layer(
                                  dimensions * 16,
                                  dimensions * 8 // factor,
                                  kernel_size = kernel_size,
                                  bias = bias,
                                  activation = activation,
                                  bilinear = bilinear
                                 )
        self.up2 = upsample_layer(
                                  dimensions * 8,
                                  dimensions * 4 // factor,
                                  kernel_size = kernel_size,
                                  bias = bias,
                                  activation = activation,
                                  bilinear = bilinear
                                 )
        self.up3 = upsample_layer(
                                  dimensions * 4,
                                  dimensions * 2 // factor,
                                  kernel_size = kernel_size,
                                  bias = bias,
                                  activation = activation,
                                  bilinear = bilinear
                                 )
        self.up4 = upsample_layer(
                                  dimensions * 2,
                                  dimensions,
                                  kernel_size = kernel_size,
                                  bias = bias,
                                  activation = activation,
                                  bilinear = bilinear
                                 )
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        result = self.outc(x9)
        return result
