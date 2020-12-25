import sys
import os
import odak
from odak import np
from odak.wave import wavenumber,plane_tilt,prism_phase_function,quadratic_phase_function

def main():
    # Variables to be set.
    wavelength                 = 0.5*pow(10,-6)
    pixeltom                   = 6*pow(10,-6)
    distance                   = 10.0
    propagation_type           = 'Fraunhofer'
    k                          = wavenumber(wavelength)
    sample_field               = np.random.rand(150,150).astype(np.complex64)
    plane_field                = plane_tilt(
                                            sample_field.shape[0],
                                            sample_field.shape[1],
                                            k,
                                            [0.3,0.9,1.,1.],
                                            dx=pixeltom,
                                            axis='x'
                                           )
    lens_field                 = quadratic_phase_function(
                                                          sample_field.shape[0],
                                                          sample_field.shape[1],
                                                          k,
                                                          focal=0.3,
                                                          dx=pixeltom
                                                         )
    prism_field                = prism_phase_function(
                                                      sample_field.shape[0],
                                                      sample_field.shape[1],
                                                      k,
                                                      angle=0.1,
                                                      dx=pixeltom,
                                                      axis='x'
                                                     )

    sample_field               = sample_field*plane_field

    #from odak.visualize.plotly import detectorshow
    #detector       = detectorshow()
    #detector.add_field(sample_field)
    #detector.show()
    #detector       = detectorshow()
    #detector.add_field(hologram)
    #detector.show()
    #detector       = detectorshow()
    #detector.add_field(reconstruction)
    #detector.show()
    assert True==True

if __name__ == '__main__':
    sys.exit(main())
