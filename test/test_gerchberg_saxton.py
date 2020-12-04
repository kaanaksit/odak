import sys

def test():
    from odak import np
    from odak.wave import gerchberg_saxton,adjust_phase_only_slm_range,produce_phase_only_slm_pattern,calculate_amplitude
    from odak.tools import save_image
    wavelength             = 0.000000532
    dx                     = 0.0000064
    distance               = 0.2
    input_field            = np.zeros((500,500),dtype=np.complex64)
    input_field[0::50,:]  += 1
    iteration_number       = 3
    hologram,reconstructed = gerchberg_saxton(
                                              input_field,
                                              iteration_number,
                                              distance,
                                              dx,
                                              wavelength,
                                              np.pi*2,
                                              'IR Fresnel'
                                             )
#    hologram               = produce_phase_only_slm_pattern(
#                                                            hologram,
#                                                            2*np.pi,
#                                                            'output_hologram.png'
#                                                           )
#    amplitude              = calculate_amplitude(reconstructed)
#    save_image(
#               'output_amplitude.png',
#               amplitude,
#               cmin=0,
#               cmax=np.amax(amplitude)
#              )
    assert True==True

if __name__ == '__main__':
    sys.exit(test())
