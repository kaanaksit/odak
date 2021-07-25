import sys

def test():
    import torch
    from odak.learn.wave import stochastic_gradient_descent
    wavelength               = 0.000000532
    dx                       = 0.0000064
    distance                 = 0.1
    resolution               = [1080,1920]
    cuda                     = False
    device                   = torch.device("cuda" if cuda else "cpu")
    input_field              = torch.zeros(resolution[0],resolution[1]).to(device)
    input_field[500::600,:]  = 1
    iteration_number         = 50
    hologram,reconstructed   = stochastic_gradient_descent(
                                                           input_field,
                                                           wavelength,
                                                           distance,
                                                           dx,
                                                           resolution,
                                                           'TR Fresnel',
                                                           iteration_number,
                                                           learning_rate=0.1,
                                                           cuda=cuda
                                                          )
    assert True==True

if __name__ == '__main__':
    sys.exit(test())
