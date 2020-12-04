import sys
import os
import odak
from odak import np
from odak.wave import wavenumber,propagate_beam,add_random_phase

def main():
    # Variables to be set.
    wavelength                 = 0.5*pow(10,-6)
    pixeltom                   = 6*pow(10,-6)
    distance                   = 0.1
    propagation_type           = 'Fraunhofer'
    k                          = wavenumber(wavelength)
    sample_field               = np.zeros((150,150),dtype=np.complex64)
    sample_field[
                 40:60,
                 40:60
                ]              = 10
    sample_field               = add_random_phase(sample_field)
    hologram                   = propagate_beam(
                                                sample_field,
                                                k,
                                                distance,
                                                pixeltom,
                                                wavelength,
                                                propagation_type
                                               )

    if propagation_type == 'Fraunhofer':
        distance = np.abs(distance)
        reconstruction             = propagate_beam(
                                                hologram,
                                                k,
                                                distance,
                                                pixeltom,
                                                wavelength,
                                                'Fraunhofer Inverse'
                                               )


    from odak.visualize.plotly import detectorshow
    detector       = detectorshow()
    detector.add_field(sample_field)
    detector.show()
    detector.add_field(hologram)
    detector.show()
    detector.add_field(reconstruction/np.amax(np.abs(reconstruction)))
    detector.show()
    assert True==True

if __name__ == '__main__':
    sys.exit(main())
