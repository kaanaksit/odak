import math
import torch
import numpy as np_cpu
import odak
from torch.functional import F
import logging



class display_color_hvs():

    def __init__(self, resolution = [1920, 1080],
                 distance_from_screen = 800,
                 pixel_pitch = 0.311,
                 read_spectrum = 'tensor',
                 primaries_spectrum = torch.rand(3, 301),
                 device = torch.device('cpu')):
        '''
        Parameters
        ----------
        resolution                  : list
                                      Resolution of the display in pixels.
        distance_from_screen        : int
                                      Distance from the screen in mm.
        pixel_pitch                 : float
                                      Pixel pitch of the display in mm.
        read_spectrum               : str
                                      Spectrum of the display. Default is 'default' which is the spectrum of the Dell U2415 display.
        device                      : torch.device
                                      Device to run the code on. Default is None which means the code will run on CPU.

        '''
        self.device = device
        self.read_spectrum = read_spectrum
        self.primaries_spectrum = primaries_spectrum.to(self.device)
        self.resolution = resolution
        self.distance_from_screen = distance_from_screen
        self.pixel_pitch = pixel_pitch
        self.l_normalised, self.m_normalised, self.s_normalised = self.initialize_cones_normalised()
        self.lms_tensor = self.construct_matrix_lms(
                                                    self.l_normalised,
                                                    self.m_normalised,
                                                    self.s_normalised
                                                   )   
        self.primaries_tensor = self.construct_matrix_primaries(
                                                    self.l_normalised,
                                                    self.m_normalised,
                                                    self.s_normalised
                                                   )   
        return
    

    def __call__(self, input_image, ground_truth, gaze=None):
        """
        Evaluating an input image against a target ground truth image for a given gaze of a viewer.
        """
        lms_image_second = self.primaries_to_lms(input_image.to(self.device))
        lms_ground_truth_second = self.primaries_to_lms(ground_truth.to(self.device))
        lms_image_third = self.second_to_third_stage(lms_image_second)
        lms_ground_truth_third = self.second_to_third_stage(lms_ground_truth_second)
        loss_metamer_color = torch.mean((lms_ground_truth_third - lms_image_third) ** 2)
        return loss_metamer_color
    
    
    def initialize_cones_normalised(self):
        """
        Internal function to initialize normalised L,M,S cones as normal distribution with given sigma, and mu values. 

        Returns
        -------
        l_cone_n                     : torch.tensor
                                       Normalised L cone distribution.
        m_cone_n                     : torch.tensor
                                       Normalised M cone distribution.
        s_cone_n                     : torch.tensor
                                       Normalised S cone distribution.
        """
        wavelength_range = np_cpu.linspace(400, 700, num=301)
        dist_l = [1 / (32.5 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (wavelength_range[i] -
                                                                               567.5)**2 / (2 * 32.5**2)) for i in range(len(wavelength_range))]
        dist_m = [1 / (27.5 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (wavelength_range[i] -
                                                                               545.0)**2 / (2 * 27.5**2)) for i in range(len(wavelength_range))]
        dist_s = [1 / (17.0 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (wavelength_range[i] -
                                                                               447.5)**2 / (2 * 17.0**2)) for i in range(len(wavelength_range))]

        l_cone_n = torch.from_numpy(dist_l/max(dist_l))
        m_cone_n = torch.from_numpy(dist_m/max(dist_m))
        s_cone_n = torch.from_numpy(dist_s/max(dist_s))
        return l_cone_n.to(self.device), m_cone_n.to(self.device), s_cone_n.to(self.device)

    
    def initialize_rgb_backlight_spectrum(self):
        """
        Internal function to initialize baclight spectrum for color primaries. 

        Returns
        -------
        red_spectrum                 : torch.tensor
                                       Normalised backlight spectrum for red color primary.
        green_spectrum               : torch.tensor
                                       Normalised backlight spectrum for green color primary.
        blue_spectrum                : torch.tensor
                                       Normalised backlight spectrum for blue color primary.
        """
        wavelength_range = np_cpu.linspace(400, 700, num=301)
        red_spectrum = [1 / (14.5 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (
            wavelength_range[i] - 650)**2 / (2 * 14.5**2)) for i in range(len(wavelength_range))]
        green_spectrum = [1 / (12 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (
            wavelength_range[i] - 550)**2 / (2 * 12.0**2)) for i in range(len(wavelength_range))]
        blue_spectrum = [1 / (12 * np_cpu.sqrt(2 * np_cpu.pi)) * np_cpu.exp(-0.5 * (
            wavelength_range[i] - 450)**2 / (2 * 12.0**2)) for i in range(len(wavelength_range))]

        red_spectrum = torch.from_numpy(
            red_spectrum / max(red_spectrum)) * 1.0
        green_spectrum = torch.from_numpy(
            green_spectrum / max(green_spectrum)) * 1.0
        blue_spectrum = torch.from_numpy(
            blue_spectrum / max(blue_spectrum)) * 1.0

        return red_spectrum.to(self.device), green_spectrum.to(self.device), blue_spectrum.to(self.device)

    
    def initialize_random_spectrum_normalised(self, dataset):
        """
        Initialize normalised light spectrum via combination of 3 gaussian distribution curve fitting [L-BFGS]. 

        Parameters
        ----------
        dataset                                : torch.tensor 
                                                 spectrum value against wavelength 
        """
        if (type(dataset).__module__) == "torch":
            dataset = dataset.numpy()
        if dataset.shape[0] > dataset.shape[1]:
            dataset = np_cpu.swapaxes(dataset, 0, 1)
        x_spectrum = np_cpu.linspace(400, 700, num=301)
        y_spectrum = np_cpu.interp(x_spectrum, dataset[0], dataset[1])
        x_spectrum = torch.from_numpy(x_spectrum) - 550
        y_spectrum = torch.from_numpy(y_spectrum)
        max_spectrum = torch.max(y_spectrum)
        y_spectrum /= max_spectrum

        def gaussian(x, A = 1, sigma = 1, centre = 0): return A * \
            torch.exp(-(x - centre) ** 2 / (2 * sigma ** 2))

        def function(x, weights): return gaussian(
            x, *weights[:3]) + gaussian(x, *weights[3:6]) + gaussian(x, *weights[6:9])
        weights = torch.tensor(
            [1.0, 1.0, -0.2, 1.0, 1.0, 0.0, 1.0, 1.0, 0.2], requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [weights], max_iter = 1000, lr = 0.1, line_search_fn=None)

        def closure():
            optimizer.zero_grad()
            output = function(x_spectrum, weights)
            loss = F.mse_loss(output, y_spectrum)
            loss.backward()
            return loss
        optimizer.step(closure)
        spectrum = function(x_spectrum, weights)
        return spectrum.detach().to(self.device)
    
   
    def display_spectrum_response(wavelength, function):
        """
        Internal function to provide light spectrum response at particular wavelength

        Parameters
        ----------
        wavelength                          : torch.tensor
                                              Wavelength in nm [400...700]
        function                            : torch.tensor
                                              Display light spectrum distribution function

        Returns
        -------
        ligth_response_dict                  : float
                                               Display light spectrum response value
        """
        wavelength = int(round(wavelength, 0))
        if wavelength >= 400 and wavelength <= 700:
            return function[wavelength - 400].item()
        elif wavelength < 400:
            return function[0].item()
        else:
            return function[300].item()

    
    def cone_response_to_spectrum(self, cone_spectrum, light_spectrum):
        """
        Internal function to calculate cone response at particular light spectrum. 

        Parameters
        ----------
        cone_spectrum                         : torch.tensor
                                                Spectrum, Wavelength [2,300] tensor 
        light_spectrum                        : torch.tensor
                                                Spectrum, Wavelength [2,300] tensor 


        Returns
        -------
        response_to_spectrum                  : float
                                                Response of cone to light spectrum [1x1] 
        """
        response_to_spectrum = torch.mul(cone_spectrum, light_spectrum)
        response_to_spectrum = torch.sum(response_to_spectrum)
        return response_to_spectrum.item()

    
    def construct_matrix_lms(self, l_response, m_response, s_response):
        '''
        Internal function to calculate cone  response at particular light spectrum. 

        Parameters
        ----------
        l_response                             : torch.tensor
                                                 Cone response spectrum tensor (normalised response vs wavelength)
        m_response                             : torch.tensor
                                                 Cone response spectrum tensor (normalised response vs wavelength)
        s_response                             : torch.tensor
                                                 Cone response spectrum tensor (normalised response vs wavelength)



        Returns
        -------
        lms_image_tensor                      : torch.tensor
                                                3x3 LMSrgb tensor

        '''
        if self.read_spectrum == 'tensor':
            logging.warning('Tensor primary spectrum is used')
            logging.warning('The number of primaries used is {}'.format(self.primaries_spectrum.shape[0]))
        else:
            logging.warning("No Spectrum data is provided")
        
        self.lms_tensor = torch.zeros(self.primaries_spectrum.shape[0], 3).to(self.device)
        for i in range(self.primaries_spectrum.shape[0]):
            self.lms_tensor[i, 0] = self.cone_response_to_spectrum(l_response,
                                                                   self.primaries_spectrum[i]
                                                                   )
            self.lms_tensor[i, 1] = self.cone_response_to_spectrum(m_response,
                                                                   self.primaries_spectrum[i]
                                                                   )
            self.lms_tensor[i, 2] = self.cone_response_to_spectrum(s_response,
                                                                   self.primaries_spectrum[i]
                                                                   ) 
        return self.lms_tensor    
    
    def construct_matrix_primaries(self, l_response, m_response, s_response):
        '''
        Internal function to calculate cone  response at particular light spectrum. 

        Parameters
        ----------
        l_response                             : torch.tensor
                                                 Cone response spectrum tensor (normalised response vs wavelength)
        m_response                             : torch.tensor
                                                 Cone response spectrum tensor (normalised response vs wavelength)
        s_response                             : torch.tensor
                                                 Cone response spectrum tensor (normalised response vs wavelength)



        Returns
        -------
        lms_image_tensor                      : torch.tensor
                                                3x3 LMSrgb tensor

        '''
        if self.read_spectrum == 'tensor':
            logging.warning('Tensor primary spectrum is used')
            logging.warning('The number of primaries used is {}'.format(self.primaries_spectrum.shape[0]))
        else:
            logging.warning("No Spectrum data is provided")
        
        self.primaries_tensor = torch.zeros(3, self.primaries_spectrum.shape[0]).to(self.device)
        for i in range(self.primaries_spectrum.shape[0]):
            self.primaries_tensor[0, i] = self.cone_response_to_spectrum(l_response,
                                                                   self.primaries_spectrum[i]
                                                                   )
            self.primaries_tensor[1, i] = self.cone_response_to_spectrum(m_response,
                                                                   self.primaries_spectrum[i]
                                                                   )
            self.primaries_tensor[2, i] = self.cone_response_to_spectrum(s_response,
                                                                   self.primaries_spectrum[i]
                                                                   ) 
        return self.primaries_tensor    
    

    def primaries_to_lms(self, primaries):
        """
        Internal function to convert primaries space to LMS space 

        Parameters
        ----------
        primaries                              : torch.tensor
                                                 Primaries data to be transformed to LMS space [BxPHxW]


        Returns
        -------
        lms_color                              : torch.tensor
                                                 LMS data transformed from Primaries space [BxPxHxW]
        """                
        primaries = primaries.permute(0, 2, 3, 1).to(self.device)
        primaries_flatten = torch.flatten(primaries, start_dim = 1, end_dim = 2)
        unflatten = torch.nn.Unflatten(1, (primaries.size(1), primaries.size(2)))
        converted_unflatten = torch.matmul(primaries_flatten.double(), self.lms_tensor.double())
        lms_color = unflatten(converted_unflatten)        
        lms_color = lms_color.permute(0, 3, 1, 2)
        return lms_color

    
    def lms_to_primaries(self, lms_color_tensor):
        """
        Internal function to convert LMS image to primaries space

        Parameters
        ----------
        lms_color_tensor                        : torch.tensor
                                                  LMS data to be transformed to primaries space [Bx3xHxW]


        Returns
        -------
        primaries                              : torch.tensor
                                               : Primaries data transformed from LMS space [BxPxHxW]
        """
        lms_color_tensor = lms_color_tensor.permute(0, 2, 3, 1).to(self.device)
        lms_color_flatten = torch.flatten(lms_color_tensor, start_dim=0, end_dim=1)
        unflatten = torch.nn.Unflatten(
            0, (lms_color_tensor.size(0), lms_color_tensor.size(1)))
        converted_unflatten = torch.matmul(
            lms_color_flatten.double(), self.lms_tensor.pinverse().double())
        primaries = unflatten(converted_unflatten)     
        primaries = primaries.permute(0, 3, 1, 2)   
        return primaries

 
    def second_to_third_stage(self, lms_image):
        '''
        This function turns second stage [L,M,S] values into third stage [(L+S)-M, M-(L+S), (M+S)-L]
        Equations are taken from Schmidt et al "Neurobiological hypothesis of color appearance and hue perception" 2014

        Parameters
        ----------
        lms_image                             : torch.tensor
                                                 Image data at LMS space (second stage)

        Returns
        -------
        third_stage                            : torch.tensor
                                                 Image data at LMS space (third stage)

        '''
        lms_image = lms_image.permute(0,2,3,1)
        third_stage = torch.zeros(lms_image.shape[0],
            lms_image.shape[1], lms_image.shape[2], 3).to(self.device)
        third_stage[:, :, :, 0] = (lms_image[:, :, :, 1] +
                                lms_image[:, :, :, 2]) - lms_image[:, :, :, 0]
        third_stage[:, :, :, 1] = (lms_image[:, :, :, 0] +
                                lms_image[:, :, :, 2]) - lms_image[:, :, :, 1]
        third_stage[:, :, :, 2] = torch.sum(lms_image, dim=3) / 3.
        third_stage = third_stage.permute(0, 3, 1, 2)
        return third_stage


    def to(self, device):
        """
        Utilization function for setting the device.
        Parameters
        ----------
        device       : torch.device
                       Device to be used (e.g., CPU, Cuda, OpenCL).
        """
        self.device = device
        return self


def rgb_2_ycrcb(image):
    """
    Converts an image from RGB colourspace to YCrCb colourspace.

    Parameters
    ----------
    image   : torch.tensor
              Input image. Should be an RGB floating-point image with values in the range [0, 1]. Should be in NCHW format [3 x m x n] or [k x 3 x m x n].

    Returns
    -------

    ycrcb   : torch.tensor
              Image converted to YCrCb colourspace [k x 3 m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
       image = image.unsqueeze(0)
    ycrcb = torch.zeros(image.size()).to(image.device)
    ycrcb[:, 0, :, :] = 0.299 * image[:, 0, :, :] + 0.587 * \
        image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
    ycrcb[:, 1, :, :] = 0.5 + 0.713 * (image[:, 0, :, :] - ycrcb[:, 0, :, :])
    ycrcb[:, 2, :, :] = 0.5 + 0.564 * (image[:, 2, :, :] - ycrcb[:, 0, :, :])
    return ycrcb


def ycrcb_2_rgb(image):
    """
    Converts an image from YCrCb colourspace to RGB colourspace.

    Parameters
    ----------
    image   : torch.tensor
              Input image. Should be a YCrCb floating-point image with values in the range [0, 1]. Should be in NCHW format [3 x m x n] or [k x 3 x m x n].

    Returns
    -------
    rgb     : torch.tensor
              Image converted to RGB colourspace [k x 3 m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
       image = image.unsqueeze(0)
    rgb = torch.zeros(image.size(), device=image.device)
    rgb[:, 0, :, :] = image[:, 0, :, :] + 1.403 * (image[:, 1, :, :] - 0.5)
    rgb[:, 1, :, :] = image[:, 0, :, :] - 0.714 * \
        (image[:, 1, :, :] - 0.5) - 0.344 * (image[:, 2, :, :] - 0.5)
    rgb[:, 2, :, :] = image[:, 0, :, :] + 1.773 * (image[:, 2, :, :] - 0.5)
    return rgb


def rgb_to_linear_rgb(image, threshold = 0.0031308):
    """
    Definition to convert RGB images to linear RGB color space. Mostly inspired from: https://kornia.readthedocs.io/en/latest/_modules/kornia/color/rgb.html

    Parameters
    ----------
    image           : torch.tensor
                      Input image in RGB color space [k x 3 x m x n] or [3 x m x n]. Image(s) must be normalized between zero and one.
    threshold       : float
                      Threshold used in calculations.

    Returns
    -------
    image_linear    : torch.tensor
                      Output image in linear RGB color space [k x 3 x m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image_linear = torch.where(image > 0.04045, torch.pow(((image + 0.055) / 1.055), 2.4), image / 12.92)
    return image_linear


def linear_rgb_to_rgb(image, threshold = 0.0031308):
    """
    Definition to convert linear RGB images to RGB color space. Mostly inspired from: https://kornia.readthedocs.io/en/latest/_modules/kornia/color/rgb.html

    Parameters
    ----------
    image           : torch.tensor
                      Input image in linear RGB color space [k x 3 x m x n] or [3 x m x n]. Image(s) must be normalized between zero and one.
    threshold       : float
                      Threshold used in calculations.

    Returns
    -------
    image_linear    : torch.tensor
                      Output image in RGB color space [k x 3 x m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image_linear =  torch.where(image > threshold, 1.055 * torch.pow(image.clamp(min=threshold), 1 / 2.4) - 0.055, 12.92 * image)
    return image_linear


def linear_rgb_to_xyz(image):
    """
    Definition to convert RGB space to CIE XYZ color space. Mostly inspired from : Rochester IT Color Conversion Algorithms (https://www.cs.rit.edu/~ncs/color/)

    Parameters
    ----------
    image           : torch.tensor
                      Input image in linear RGB color space [k x 3 x m x n] or [3 x m x n]. Image(s) must be normalized between zero and one.

    Returns
    -------
    image_xyz       : torch.tensor
                      Output image in XYZ (CIE 1931) color space [k x 3 x m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    a11 = 0.412453
    a12 = 0.357580
    a13 = 0.180423
    a21 = 0.212671
    a22 = 0.715160
    a23 = 0.072169
    a31 = 0.019334
    a32 = 0.119193
    a33 = 0.950227
    M = torch.tensor([[a11, a12, a13], 
                      [a21, a22, a23],
                      [a31, a32, a33]])
    size = image.size()
    image = image.reshape(size[0], size[1], size[2]*size[3])  # NC(HW)
    image_xyz = torch.matmul(M, image)
    image_xyz = image_xyz.reshape(size[0], size[1], size[2], size[3])
    return image_xyz


def xyz_to_linear_rgb(image):
    """
    Definition to convert CIE XYZ space to linear RGB color space. Mostly inspired from : Rochester IT Color Conversion Algorithms (https://www.cs.rit.edu/~ncs/color/)

    Parameters
    ----------
    image            : torch.tensor
                       Input image in XYZ (CIE 1931) color space [k x 3 x m x n] or [3 x m x n]. Image(s) must be normalized between zero and one.

    Returns
    -------
    image_linear_rgb : torch.tensor
                       Output image in linear RGB  color space [k x 3 x m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    a11 = 3.240479
    a12 = -1.537150
    a13 = -0.498535
    a21 = -0.969256 
    a22 = 1.875992 
    a23 = 0.041556
    a31 = 0.055648
    a32 = -0.204043
    a33 = 1.057311
    M = torch.tensor([[a11, a12, a13], 
                      [a21, a22, a23],
                      [a31, a32, a33]])
    size = image.size()
    image = image.reshape(size[0], size[1], size[2]*size[3])
    image_linear_rgb = torch.matmul(M, image)
    image_linear_rgb = image_linear_rgb.reshape(size[0], size[1], size[2], size[3])
    return image_linear_rgb

def rgb_to_hsv(image, eps: float = 1e-8):
    
    """
    Definition to convert RGB space to HSV color space. Mostly inspired from : https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html

    Parameters
    ----------
    image           : torch.tensor
                      Input image in HSV color space [k x 3 x m x n] or [3 x m x n]. Image(s) must be normalized between zero and one.

    Returns
    -------
    image_hsv       : torch.tensor
                      Output image in  RGB  color space [k x 3 x m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac = max_rgb - min_rgb
    v = max_rgb
    s = deltac / (max_rgb + eps)
    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)
    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac
    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0
    h = 2.0 * math.pi * h 
    image_hsv = torch.stack((h, s, v), dim=-3)
    return image_hsv


def hsv_to_rgb(image):
    
    """
    Definition to convert HSV space to  RGB color space. Mostly inspired from : https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html

    Parameters
    ----------
    image           : torch.tensor
                      Input image in HSV color space [k x 3 x m x n] or [3 x m x n]. Image(s) must be normalized between zero and one.

    Returns
    -------
    image_rgb       : torch.tensor
                      Output image in  RGB  color space [k x 3 x m x n] or [1 x 3 x m x n].
    """
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    h = image[..., 0, :, :] / (2 * math.pi)
    s = image[..., 1, :, :]
    v = image[..., 2, :, :]
    hi = torch.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi
    one = torch.tensor(1.0)
    p = v * (one - s)
    q = v * (one - f * s)
    t = v * (one - (one - f) * s)
    hi = hi.long()
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    image_rgb = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    image_rgb = torch.gather(image_rgb, -3, indices)
    return image_rgb


def srgb_to_lab(image):    
    """
    Definition to convert SRGB space to LAB color space. 

    Parameters
    ----------
    image           : torch.tensor
                      Input image in SRGB color space[3 x m x n]
    Returns
    -------
    image_lab       : torch.tensor
                      Output image in LAB color space [3 x m x n].
    """
    if image.shape[-1] == 3:
        input_color = image.permute(2, 0, 1)  # C(H*W)
    else:
        input_color = image
    # rgb ---> linear rgb
    limit = 0.04045        
    # linear rgb ---> xyz
    linrgb_color = torch.where(input_color > limit, torch.pow((input_color + 0.055) / 1.055, 2.4), input_color / 12.92)

    a11 = 10135552 / 24577794
    a12 = 8788810  / 24577794
    a13 = 4435075  / 24577794
    a21 = 2613072  / 12288897
    a22 = 8788810  / 12288897
    a23 = 887015   / 12288897
    a31 = 1425312  / 73733382
    a32 = 8788810  / 73733382
    a33 = 70074185 / 73733382

    A = torch.tensor([[a11, a12, a13],
                    [a21, a22, a23],
                    [a31, a32, a33]], dtype=torch.float32)

    linrgb_color = linrgb_color.permute(2, 0, 1) # C(H*W)
    xyz_color = torch.matmul(A, linrgb_color)
    xyz_color = xyz_color.permute(1, 2, 0)
    # xyz ---> lab
    inv_reference_illuminant = torch.tensor([[[1.052156925]], [[1.000000000]], [[0.918357670]]], dtype=torch.float32)
    input_color = xyz_color * inv_reference_illuminant
    delta = 6 / 29
    delta_square = delta * delta
    delta_cube = delta * delta_square
    factor = 1 / (3 * delta_square)

    input_color = torch.where(input_color > delta_cube, torch.pow(input_color, 1 / 3), (factor * input_color + 4 / 29))

    l = 116 * input_color[1:2, :, :] - 16
    a = 500 * (input_color[0:1,:, :] - input_color[1:2, :, :])
    b = 200 * (input_color[1:2, :, :] - input_color[2:3, :, :])

    image_lab = torch.cat((l, a, b), 0)
    return image_lab    

def lab_to_srgb(image):
    """
    Definition to convert LAB space to SRGB color space. 

    Parameters
    ----------
    image           : torch.tensor
                      Input image in LAB color space[3 x m x n]
    Returns
    -------
    image_srgb     : torch.tensor
                      Output image in SRGB color space [3 x m x n].
    """
    
    if image.shape[-1] == 3:
        input_color = image.permute(2, 0, 1)  # C(H*W)
    else:
        input_color = image
    # lab ---> xyz
    reference_illuminant = torch.tensor([[[0.950428545]], [[1.000000000]], [[1.088900371]]], dtype=torch.float32)
    y = (input_color[0:1, :, :] + 16) / 116
    a =  input_color[1:2, :, :] / 500
    b =  input_color[2:3, :, :] / 200
    x = y + a
    z = y - b
    xyz = torch.cat((x, y, z), 0)
    delta = 6 / 29
    factor = 3 * delta * delta
    xyz = torch.where(xyz > delta,  xyz ** 3, factor * (xyz - 4 / 29))
    xyz_color = xyz * reference_illuminant
    # xyz ---> linear rgb
    a11 = 3.241003275
    a12 = -1.537398934
    a13 = -0.498615861
    a21 = -0.969224334
    a22 = 1.875930071
    a23 = 0.041554224
    a31 = 0.055639423
    a32 = -0.204011202
    a33 = 1.057148933
    A = torch.tensor([[a11, a12, a13],
                  [a21, a22, a23],
                  [a31, a32, a33]], dtype=torch.float32)

    xyz_color = xyz_color.permute(2, 0, 1) # C(H*W)
    linear_rgb_color = torch.matmul(A, xyz_color)
    linear_rgb_color = linear_rgb_color.permute(1, 2, 0)
    # linear rgb ---> srgb
    limit = 0.0031308
    image_srgb = torch.where(linear_rgb_color > limit, 1.055 * (linear_rgb_color ** (1.0 / 2.4)) - 0.055, 12.92 * linear_rgb_color)
    return image_srgb

def color_map(input_image, target_image, model = 'Lab Stats'):
    """
    Internal function to map the color of an image to another image.
    Reference: Color transfer between images, Reinhard et al., 2001.
    
    Parameters
    ----------
    input_image         : torch.Tensor
                          Input image in RGB color space [3 x m x n].
    target_image        : torch.Tensor
    
    Returns
    -------
    mapped_image           : torch.Tensor
                             Input image with the color the distribution of the target image [3 x m x n].
    """
    if model == 'Lab Stats':
        lab_input = srgb_to_lab(input_image)
        lab_target = srgb_to_lab(target_image)
        input_mean_L = torch.mean(lab_input[0, :, :])
        input_mean_a = torch.mean(lab_input[1, :, :])
        input_mean_b = torch.mean(lab_input[2, :, :])
        input_std_L = torch.std(lab_input[0, :, :])
        input_std_a = torch.std(lab_input[1, :, :])
        input_std_b = torch.std(lab_input[2, :, :])
        target_mean_L = torch.mean(lab_target[0, :, :])
        target_mean_a = torch.mean(lab_target[1, :, :])
        target_mean_b = torch.mean(lab_target[2, :, :])
        target_std_L = torch.std(lab_target[0, :, :])
        target_std_a = torch.std(lab_target[1, :, :])
        target_std_b = torch.std(lab_target[2, :, :])
        lab_input[0, :, :] = (lab_input[0, :, :] - input_mean_L) * (target_std_L / input_std_L) + target_mean_L
        lab_input[1, :, :] = (lab_input[1, :, :] - input_mean_a) * (target_std_a / input_std_a) + target_mean_a
        lab_input[2, :, :] = (lab_input[2, :, :] - input_mean_b) * (target_std_b / input_std_b) + target_mean_b
        mapped_image = lab_to_srgb(lab_input.permute(1, 2, 0))
        return mapped_image
