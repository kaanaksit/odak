import sys
import os
import odak
from odak import np
from odak.wave import wavenumber,propagate_beam
from odak.learn import propagate_beam as propagate_beam_torch
import torch

def main():
    # Variables to be set.
    wavelength                 = 0.5*pow(10,-6)
    pixeltom                   = 6*pow(10,-6)
    distance                   = 0.2
    propagation_type           = 'IR Fresnel'
    k                          = wavenumber(wavelength)
    sample_field               = np.zeros((500,500),dtype=np.complex64)
    sample_field[
                 240:260,
                 240:260
                ]              = 1000
    random_phase               = np.pi*np.random.random(sample_field.shape)
    sample_field               = sample_field*np.cos(random_phase)+1j*sample_field*np.sin(random_phase)
    sample_field               = torch.from_numpy(sample_field)
    hologram                   = propagate_beam_torch(
                                                sample_field,
                                                k,
                                                distance,
                                                pixeltom,
                                                wavelength,
                                                propagation_type
                                               )
    reconstruction             = propagate_beam_torch(
                                                hologram,
                                                k,
                                                -distance,
                                                pixeltom,
                                                wavelength,
                                                propagation_type
                                               )
    # from odak.visualize.plotly import detectorshow
    # detector       = detectorshow()
    # detector.add_field(sample_field)
    # detector.show()
    # detector.add_field(hologram)
    # detector.show()
    # detector.add_field(reconstruction)
    # detector.show()
    assert True==True

def compare():
    wavelength                 = 0.5*pow(10,-6)
    pixeltom                   = 6*pow(10,-6)
    distance                   = 0.2
    propagation_type           = 'IR Fresnel'
    k                          = wavenumber(wavelength)
    sample_field               = np.zeros((500,500),dtype=np.complex64)
    sample_field[
                 240:260,
                 240:260
                ]              = 1000
    random_phase               = np.pi*np.random.random(sample_field.shape)
    sample_field               = sample_field*np.cos(random_phase)+1j*sample_field*np.sin(random_phase)
    
    sample_field_torch         = torch.from_numpy(sample_field)

    ## Propagate and reconstruct using torch.
    hologram_torch                   = propagate_beam_torch(
                                                sample_field_torch,
                                                k,
                                                distance,
                                                pixeltom,
                                                wavelength,
                                                propagation_type
                                               )
    reconstruction_torch             = propagate_beam_torch(
                                                hologram_torch,
                                                k,
                                                -distance,
                                                pixeltom,
                                                wavelength,
                                                propagation_type
                                               )   

    ## Propagate and reconstruct using np.
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

    np.testing.assert_array_almost_equal(hologram_torch.numpy(), hologram, 3)

if __name__ == '__main__':
    sys.exit(main())
