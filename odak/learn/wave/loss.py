import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class phase_gradient(nn.Module):
    
    """
    The class 'phase_gradient' provides a regularization function to measure the variation(Gradient or Laplace) of the phase of the complex amplitude. 

    This implements a convolution of the phase with a kernel.

    The kernel is a simple 3 by 3 Laplacian kernel here, but you can also try other edge detection methods.
    """
    

    def __init__(self, kernel = None, loss = nn.MSELoss(), device=torch.device("cpu")):
        """
        Parameters
        ----------
        kernel                  : torch.tensor
                                    Convolution filter kernel, 3 by 3 Laplacian kernel by default.
        loss                    : torch.nn.Module
                                    loss function, L2 Loss by default.
        """
        super(phase_gradient, self).__init__()
        self.device = device
        self.loss = loss
        if kernel == None:
            self.kernel = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32)/8
        else:
            if len(kernel.shape) == 4:
                self.kernel = kernel
            else:
                self.kernel = kernel.reshape((1, 1, kernel.shape[0], kernel.shape[1]))
        self.kernel = Variable(self.kernel.to(self.device))
        

    def forward(self, phase):
        """
        Calculates the phase gradient Loss.

        Parameters
        ----------
        phase                  : torch.tensor
                                    Phase of the complex amplitude.

        Returns
        -------

        loss_value              : torch.tensor
                                    The computed loss.
        """

        if len(phase.shape) == 2:
            phase = phase.reshape((1, 1, phase.shape[0], phase.shape[1]))
        edge_detect = self.functional_conv2d(phase)
        loss_value = self.loss(edge_detect, torch.zeros_like(edge_detect))
        return loss_value

    def functional_conv2d(self, phase):
        """
        Calculates the gradient of the phase.

        Parameters
        ----------
        phase                  : torch.tensor
                                    Phase of the complex amplitude.

        Returns
        -------

        edge_detect              : torch.tensor
                                    The computed phase gradient.
        """
        edge_detect = F.conv2d(phase, self.kernel, padding=self.kernel.shape[-1]//2)
        return edge_detect



class speckle_contrast(nn.Module):

    """
    The class 'speckle_contrast' provides a regularization function to measure the speckle contrast of the intensity of the complex amplitude using C=sigma/mean. Where C is the speckle contrast, mean and sigma are mean and standard deviation of the intensity.

    We refer to the following paper:

    Kim et al.(2020). Light source optimization for partially coherent holographic displays with consideration of speckle contrast, resolution, and depth of field. Scientific Reports. 10. 18832. 10.1038/s41598-020-75947-0. 
    """
    

    def __init__(self, kernel_size = 11, step_size = (1, 1), loss = nn.MSELoss(), device=torch.device("cpu")):
        """
        Parameters
        ----------
        kernel_size             : torch.tensor
                                    Convolution filter kernel size, 11 by 11 average kernel by default.
        step_size               : tuple
                                    Convolution stride in height and width direction.
        loss                    : torch.nn.Module
                                    loss function, L2 Loss by default.
        """
        super(speckle_contrast, self).__init__()
        self.device = device
        self.loss = loss
        self.step_size = step_size
        self.kernel_size = kernel_size
        self.kernel = torch.ones((1, 1, self.kernel_size, self.kernel_size))/(self.kernel_size**2)
        self.kernel = Variable(self.kernel.type(torch.FloatTensor).to(self.device))


    def forward(self, intensity):
        """
        Calculates the speckle contrast Loss.

        Parameters
        ----------
        intensity               : torch.tensor
                                    intensity of the complex amplitude.

        Returns
        -------

        loss_value              : torch.tensor
                                    The computed loss.
        """

        if len(intensity.shape) == 2:
            intensity = intensity.reshape((1, 1, intensity.shape[0], intensity.shape[1]))
        Speckle_C = self.functional_conv2d(intensity)
        loss_value = self.loss(Speckle_C, torch.zeros_like(Speckle_C))
        return loss_value


    def functional_conv2d(self, intensity):
        """
        Calculates the speckle contrast of the intensity.

        Parameters
        ----------
        intensity                : torch.tensor
                                    Intensity of the complex field.

        Returns
        -------

        Speckle_C               : torch.tensor
                                    The computed speckle contrast.
        """
        mean = F.conv2d(intensity, self.kernel, stride = self.step_size)
        var = torch.sqrt(F.conv2d(torch.pow(intensity, 2), self.kernel, stride = self.step_size) - torch.pow(mean, 2))
        Speckle_C = var/mean
        return Speckle_C
