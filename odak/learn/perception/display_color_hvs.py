import wave
import numpy as np_cpu
import torch
import odak
from torch.functional import F


class DisplayColorHVS():

    def __init__(self, resolution=[1920, 1080],
                 distance_from_screen=800,
                 pixel_pitch=0.311,
                 read_spectrum='backlight',
                 spectrum_data_root = './backlight',
                 device=None):
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
        spectrum_data_root          : str
                                      Path to the folder containing the spectrum data of the display.
        device                      : torch.device
                                      Device to run the code on. Default is None which means the code will run on CPU.

        '''
        self.device = device
        if isinstance(self.device, type(None)):
            self.device = torch.device("cpu")
        self.read_spectrum = read_spectrum
        self.resolution = resolution
        self.distance_from_screen = distance_from_screen
        self.pixel_pitch = pixel_pitch
        self.spectrum_data_root = spectrum_data_root
        self.l_normalised, self.m_normalised, self.s_normalised = self.initialise_cones_normalised()
        self.lms_tensor = self.construct_matrix_lms(
                                                    self.l_normalised,
                                                    self.m_normalised,
                                                    self.s_normalised
                                                   )   
        return
    

    def __call__(self, input_image, ground_truth, gaze=None):
        """
        Evaluating an input image against a target ground truth image for a given gaze of a viewer.
        """
        lms_image_second = self.rgb_to_lms(input_image.to(self.device))
        lms_ground_truth_second = self.rgb_to_lms(ground_truth.to(self.device))
        lms_image_third = self.second_to_third_stage(lms_image_second)
        lms_ground_truth_third = self.second_to_third_stage(
            lms_ground_truth_second)
        loss_metamer_color = torch.mean(
            (lms_ground_truth_third - lms_image_third)**2)

        return loss_metamer_color
    
    
    def initialise_cones_normalised(self):
        """
        Internal function to nitialise normalised L,M,S cones as normal distribution with given sigma, and mu values. 

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

    
    def initialise_rgb_backlight_spectrum(self):
        """
        Internal function to initialise baclight spectrum for color primaries. 

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

    
    def initialise_random_spectrum_normalised(self, dataset):
        """
        Initialise normalised light spectrum via combination of 3 gaussian distribution curve fitting [L-BFGS]. 
        Parameters
        ----------
        dataset                                : torch.tensor 
                                                 spectrum value against wavelength 
        peakspectrum                           :

        Returns
        -------
        light_spectrum                         : torch.tensor
                                                 Normalized light spectrum function       
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

        def gaussian(x, A=1, sigma=1, centre=0): return A * \
            torch.exp(-(x - centre)**2 / (2*sigma**2))

        def function(x, weights): return gaussian(
            x, *weights[:3]) + gaussian(x, *weights[3:6]) + gaussian(x, *weights[6:9])
        weights = torch.tensor(
            [1.0, 1.0, -0.2, 1.0, 1.0, 0.0, 1.0, 1.0, 0.2], requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [weights], max_iter=1000, lr=0.1, line_search_fn=None)

        def closure():
            optimizer.zero_grad()
            output = function(x_spectrum, weights)
            loss = F.mse_loss(output, y_spectrum)
            loss.backward()
            return loss
        optimizer.step(closure)
        spectrum = function(x_spectrum, weights)
        return spectrum.detach().to(self.device)

    
    def initialise_normalised_spectrum_primaries(self):
        '''
        Initialise normalised light spectrum via csv data and multilayer perceptron curve fitting. 

        Returns
        -------
        red_spectrum_fit                         : torch.tensor
                                                   Fitted red light spectrum function 
        green_spectrum_fit                       : torch.tensor
                                                   Fitted green light spectrum function  
        blue_spectrum_fit                        : torch.tensor
                                                   Fitted blue light spectrum function
        '''
        import os
        root = self.spectrum_data_root
        print("Reading the display spectrum data from {}".format(root))
        red_data = np_cpu.swapaxes(np_cpu.genfromtxt(
            root + 'red_spectrum.csv', delimiter=','), 0, 1)
        green_data = np_cpu.swapaxes(np_cpu.genfromtxt(
             root + 'green_spectrum.csv', delimiter=','), 0, 1)
        blue_data = np_cpu.swapaxes(np_cpu.genfromtxt(
             root + 'blue_spectrum.csv', delimiter=','), 0, 1)
        wavelength = np_cpu.linspace(400, 700, num=301)
        red_spectrum = torch.from_numpy(np_cpu.interp(
            wavelength, red_data[0], red_data[1])).unsqueeze(1)
        green_spectrum = torch.from_numpy(np_cpu.interp(
            wavelength, green_data[0], green_data[1])).unsqueeze(1)
        blue_spectrum = torch.from_numpy(np_cpu.interp(
            wavelength, blue_data[0], blue_data[1])).unsqueeze(1)
        wavelength = torch.from_numpy(wavelength).unsqueeze(1) / 550.
        curve = odak.learn.tools.multi_layer_perceptron(
            n_hidden=64).to(torch.device('cpu'))  # device
        curve.fit(wavelength.float(), red_spectrum.float(),
                  epochs=1000, learning_rate=5e-4)
        red_estimate = torch.zeros_like(red_spectrum)
        for i in range(red_estimate.shape[0]):
            red_estimate[i] = curve.forward(wavelength[i].float().view(1, 1))
        curve.fit(wavelength.float(), green_spectrum.float(),
                  epochs=1000, learning_rate=5e-4)
        green_estimate = torch.zeros_like(green_spectrum)
        for i in range(green_estimate.shape[0]):
            green_estimate[i] = curve.forward(wavelength[i].float().view(1, 1))
        curve.fit(wavelength.float(), blue_spectrum.float(),
                  epochs=1000, learning_rate=5e-4)
        blue_estimate = torch.zeros_like(blue_spectrum)
        for i in range(blue_estimate.shape[0]):
            blue_estimate[i] = curve.forward(wavelength[i].float().view(1, 1))
        primary_wavelength = torch.linspace(400, 700, 301)
        red_spectrum_fit = torch.cat(
            (primary_wavelength.unsqueeze(1), red_estimate), 1)
        green_spectrum_fit = torch.cat(
            (primary_wavelength.unsqueeze(1), green_estimate), 1)
        blue_spectrum_fit = torch.cat(
            (primary_wavelength.unsqueeze(1), blue_estimate), 1)
        red_spectrum_fit[:, 1] *= (red_spectrum_fit[:, 1] > 0)
        green_spectrum_fit[:, 1] *= (green_spectrum_fit[:, 1] > 0)
        blue_spectrum_fit[:, 1] *= (blue_spectrum_fit[:, 1] > 0)
        red_spectrum_fit=red_spectrum_fit.detach()
        green_spectrum_fit=green_spectrum_fit.detach()
        blue_spectrum_fit=blue_spectrum_fit.detach()
        return red_spectrum_fit[:, 1].to(self.device), green_spectrum_fit[:, 1].to(self.device), blue_spectrum_fit[:, 1].to(self.device)

    
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
        *_response                             : torch.tensor
                                                 Cone response spectrum tensor (normalised response vs wavelength)

        Returns
        -------
        lms_image_tensor                      : torch.tensor
                                                3x3 LMSrgb tensor

        '''
        if self.read_spectrum == 'backlight':
            print('*.csv backlight data is used ')
            red_spectrum, green_spectrum, blue_spectrum = self.initialise_normalised_spectrum_primaries()
        else:
            print('Backlight data is not provided, estimated gaussian backlight is used')
            red_spectrum, green_spectrum, blue_spectrum = self.initialise_rgb_backlight_spectrum()

        l_r = self.cone_response_to_spectrum(l_response, red_spectrum)
        l_g = self.cone_response_to_spectrum(l_response, green_spectrum)
        l_b = self.cone_response_to_spectrum(l_response, blue_spectrum)
        m_r = self.cone_response_to_spectrum(m_response, red_spectrum)
        m_g = self.cone_response_to_spectrum(m_response, green_spectrum)
        m_b = self.cone_response_to_spectrum(m_response, blue_spectrum)
        s_r = self.cone_response_to_spectrum(s_response, red_spectrum)
        s_g = self.cone_response_to_spectrum(s_response, green_spectrum)
        s_b = self.cone_response_to_spectrum(s_response, blue_spectrum)
        self.lms_tensor = torch.tensor(
            [[l_r, m_r, s_r], [l_g, m_g, s_g], [l_b, m_b, s_b]]).to(self.device)
        return self.lms_tensor      


    def rgb_to_lms(self, rgb_image_tensor):
        """
        Internal function to convert RGB image to LMS space 

        Parameters
        ----------
        rgb_image_tensor                      : torch.tensor
                                                Image RGB data to be transformed to LMS space


        Returns
        -------
        lms_image_tensor                      : float
                                              : Image LMS data transformed from RGB space [3xHxW]
        """
        image_flatten = torch.flatten(rgb_image_tensor, start_dim=0, end_dim=1)
        unflatten = torch.nn.Unflatten(
            0, (rgb_image_tensor.size(0), rgb_image_tensor.size(1)))
        converted_unflatten = torch.matmul(
            image_flatten.double(), self.lms_tensor.double())
        converted_image = unflatten(converted_unflatten)        
        return converted_image.to(self.device)

    
    def lms_to_rgb(self, lms_image_tensor):
        """
        Internal function to convert LMS image to RGB space

        Parameters
        ----------
        lms_image_tensor                      : torch.tensor
                                                Image LMS data to be transformed to RGB space


        Returns
        -------
       rgb_image_tensor                       : float
                                              : Image RGB data transformed from RGB space [3xHxW]
        """
        image_flatten = torch.flatten(lms_image_tensor, start_dim=0, end_dim=1)
        unflatten = torch.nn.Unflatten(
            0, (lms_image_tensor.size(0), lms_image_tensor.size(1)))
        converted_unflatten = torch.matmul(
            image_flatten.double(), self.lms_tensor.inverse().double())
        converted_rgb_image = unflatten(converted_unflatten)        
        return converted_rgb_image.to(self.device)


    def convert_to_lms(self, image_channel, wavelength, intensity):
        """
        Internal function to convert color primary to LMS space

        Parameters
        ----------
        image_channel                         : torch.tensor
                                                Image color primary channel data to be transformed to LMS space.
        wavelength                            : float 
                                                Particular wavelength to be used in LMS conversion.
        intensity                             : float
                                                Particular intensity of color primary spectrum with respect to wavelength to be used in LMS conversion.


        Returns
        -------
        lms_image                             : torch.tensor 
                                                Image channel LMS data transformed from color primary to LMS space [HxWx3].
        """
        spectrum = torch.zeros(301)
        spectrum[wavelength-400] = intensity 
        l = self.cone_response_to_spectrum(self.l_normalised, spectrum)
        m = self.cone_response_to_spectrum(self.m_normalised, spectrum)
        s = self.cone_response_to_spectrum(self.s_normalised, spectrum)
        lms_tensor_wavelength = torch.tensor([l, m, s]).to(self.device)
        image_flatten = torch.flatten(image_channel, start_dim=0, end_dim=1)
        image_flatten = image_flatten.unsqueeze(0).swapaxes(0, 1)
        lms_tensor_wavelength = lms_tensor_wavelength.unsqueeze(0)
        lms_converted_flatten = torch.matmul(
            image_flatten.double(), lms_tensor_wavelength.double())
        unflatten = torch.nn.Unflatten(
            0, (image_channel.size(0), image_channel.size(1)))
        lms_image = unflatten(lms_converted_flatten)
        return lms_image.to(self.device)

 
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
        third_stage = torch.zeros(
            lms_image.shape[0], lms_image.shape[1], 3).to(self.device)
        third_stage[:, :, 0] = (lms_image[:, :, 1] +
                                lms_image[:, :, 2]) - lms_image[:, :, 0]
        third_stage[:, :, 1] = (lms_image[:, :, 0] +
                                lms_image[:, :, 2]) - lms_image[:, :, 1]
        third_stage[:, :, 2] = torch.sum(lms_image, dim=2) / 3.
        return third_stage


    def convert_to_second_stage(self,image_channel, wavelength, intensity):
       second_stage_image = self.second_to_third_stage(self.convert_to_lms(image_channel, wavelength, intensity))
       return second_stage_image.to(self.device)


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