import sys

def test():
    from odak import np
    import torch
    from odak.learn.wave import gerchberg_saxton,produce_phase_only_slm_pattern,calculate_amplitude
    from odak.tools import save_image
    wavelength             = 0.000000532
    dx                     = 0.0000064
    distance               = 0.2
    input_field            = np.zeros((500,500),dtype=np.complex64)
    input_field[0::50,:]  += 1
    iteration_number       = 3
    if np.__name__ == 'cupy':
        input_field = np.asnumpy(input_field)
    input_field            = torch.from_numpy(input_field)
    hologram,reconstructed = gerchberg_saxton(
                                              input_field,
                                              iteration_number,
                                              distance,
                                              dx,
                                              wavelength,
                                              np.pi*2,
                                              'TR Fresnel'
                                             )
    # hologram,_             = produce_phase_only_slm_pattern(
    #                                                         hologram,
    #                                                         2*np.pi
    #                                                         )
    # amplitude              = calculate_amplitude(reconstructed)
    # amplitude              = amplitude.numpy()
    # save_image(
    #             'output_amplitude_torch.png',
    #             amplitude,
    #             cmin=0,
    #             cmax=np.amax(amplitude)
    #             )
    assert True==True

if __name__ == '__main__':
    sys.exit(test())
