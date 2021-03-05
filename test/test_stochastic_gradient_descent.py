import sys

def test():
    from odak import np
    from tqdm import tqdm
    import torch.optim as optim
    import torch.nn as nn
    import torch
    import odak.tools as tools
    from odak.learn.wave import stochastic_gradient_descent, calculate_phase, calculate_amplitude
    import odak.learn.tools as learntools
    wavelength               = 0.000000532
    dx                       = 0.000008
    resolution               = [1080,1920]
    distance                 = 0.15
    propogation_type         = "TR Fresnel"
    input_field              = np.zeros((1080,1920),dtype=np.float)
    input_field[0::50,:]    += 1
    iteration_number         = 10
    input_field              = tools.convert_to_torch(input_field)
    hologram, reconstruction = stochastic_gradient_descent(
                                                           input_field, 
                                                           wavelength, 
                                                           distance, 
                                                           dx, 
                                                           resolution, 
                                                           propogation_type, 
                                                           iteration_number,
                                                           loss_function=None, 
                                                           cuda=False
                                                           )
    phase                    = ((calculate_phase(hologram) % (2 * np.pi)) / (2*np.pi)) * 255 
    phase                    = np.asarray(tools.convert_to_numpy(phase))
    reconstruction_intensity = calculate_amplitude(reconstruction).float()
    reconstruction_intensity = np.asarray(tools.convert_to_numpy(reconstruction_intensity))
    reconstruction_intensity = reconstruction_intensity/np.amax(reconstruction_intensity) * 255

if __name__ == "__main__":
    sys.exit(test())
