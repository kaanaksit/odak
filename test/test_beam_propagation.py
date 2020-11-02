import sys
import os
import odak
from odak import np
from odak.wave import wavenumber,propagate_beam

def main():
    # Variables to be set.
    wavelength                 = 0.5*pow(10,-6)
    pixeltom                   = 3*pow(10,-6)
    distance                   = 0.2
    propagation_type           = 'IR Fresnel'
    k                          = wavenumber(wavelength)
    sample_field               = np.zeros((500,500),dtype=np.complex64)
    sample_field[
                 240:260,
                 240:260
                ]              = 1000
    sample_field              += 1j*np.random.random(sample_field.shape)
    hologram                   = propagate_beam(
                                                sample_field,
                                                k,
                                                distance,
                                                pixeltom,
                                                wavelength,
                                                propagation_type
                                               )
    reconstruction             = propagate_beam(
                                                hologram,
                                                k,
                                                -distance,
                                                pixeltom,
                                                wavelength,
                                                propagation_type
                                               )
#    from odak.visualize.plotly import detectorshow
#    detector       = detectorshow()
#    detector.add_field(sample_field)
#    detector.show()
#    detector.add_field(hologram)
#    detector.show()
#    detector.add_field(reconstruction)
#    detector.show()
    assert True==True

if __name__ == '__main__':
    sys.exit(main())
