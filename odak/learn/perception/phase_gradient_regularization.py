import torch
import torch.nn.functional as F
from math import exp
import torch.nn as nn
import numpy as np
from torch.autograd import Variable



class phase_gradient(nn.Module):
    
    """
    A regularization function to measure the variation(Gradient or Laplace) of the phase of the complex amplitude. 
    We use a simple 3 by 3 Laplacian kernel here, but you can also try other edge detection methods.

    Input
    -----
    :param kernel: Convolution filter kernel, 3 by 3 Laplacian kernel by default.
    :param loss: loss function, L2 Loss by default.
    :param device: GPU or CPU, GPU by default.

    Output
    -----
    :return: loss_value: a scalar loss value.
    """

    def __init__(self, kernel = None, loss = nn.MSELoss(), device = torch.device('cuda')):
        super(phase_gradient, self).__init__()
        self.device = device
        self.loss = loss
        if kernel == None:
            self.kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')/8
        else:
            self.kernel = kernel
        self.kernel = self.kernel.reshape((1, 1, self.kernel.shape[0], self.kernel.shape[1]))
        self.kernel = Variable(torch.from_numpy(self.kernel).to(self.device))
        

    def forward(self, phase):
        edge_detect = self.functional_conv2d(phase)
        loss_value = self.loss(edge_detect, torch.zeros_like(edge_detect))
        return loss_value

    def functional_conv2d(self, phase):
        edge_detect = F.conv2d(phase, self.kernel, padding=self.kernel.shape[-1])
        return edge_detect



class speckle_contrast(nn.Module):

    """
    A regularization function to measure the speckle contrast of the intensity of the complex amplitude using C=sigma/mean. 
    Where C is the speckle contrast, mean and sigma are mean and standard deviation of the intensity.
    We refer to the following paper:
    Kim et al.(2020). Light source optimization for partially coherent holographic displays with consideration of speckle contrast, 
    resolution, and depth of field. Scientific Reports. 10. 18832. 10.1038/s41598-020-75947-0. 
    But I think this function may not be more effective than a simple MSE loss.

    Input
    -----
    :param kernel: Convolution filter kernel, 11 by 11 average kernel by default.
    :step_size: Convolution stride in height and width direction.
    :param loss: loss function, L2 Loss by default.
    :param device: GPU or CPU, GPU by default.

    Output
    -----
    :return: loss_value: a scalar loss value.
    """

    def __init__(self, kernel_size = 11, step_size = (1, 1), loss = nn.MSELoss(), device = torch.device('cuda')):
        super(speckle_contrast, self).__init__()
        self.device = device
        self.loss = loss
        self.step_size = step_size
        self.kernel_size = kernel_size
        self.kernel = np.ones((self.kernel_size, self.kernel_size))/(self.kernel_size**2)
        self.kernel = self.kernel.reshape((1, 1, self.kernel.shape[0], self.kernel.shape[1]))
        self.kernel = Variable(torch.from_numpy(self.kernel).type(torch.FloatTensor).to(self.device))


    def forward(self, intensity):
        Speckle_C = self.functional_conv2d(intensity)
        loss_value = self.loss(Speckle_C, torch.zeros_like(Speckle_C))
        return loss_value


    def functional_conv2d(self, intensity):
        mean = F.conv2d(intensity, self.kernel, stride = self.step_size)
        var = torch.sqrt(F.conv2d(intensity ** 2, self.kernel, stride = self.step_size) - mean**2)
        Speckle_C = var/mean
        return Speckle_C