#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,odak

__author__  = ('Kaan Akşit')

def example():
    # Fixed values are set
    onepxtom     = pow(10,-5)
    distance     = 0.3
    wavelength   = 500*pow(10,-9)
    aperturesize = 40
    pxx          = 128
    pxy          = 128
    diffrac      = odak.diffractions()
    aperture     = odak.aperture()
    beam         = odak.beams()
    # Defining the aperture
    rectangle    = aperture.rectangle(pxx,pxy,aperturesize)
    gaussian     = aperture.gaussian(pxx,pxy,aperturesize)
    circle       = aperture.circle(pxx,pxy,aperturesize)
    #aperture.show(rectangle,onepxtom,wavelength,'Aperture')
    # Defining a diverging spherical wave
    focal        = 0.0001
    for distance in xrange(1,10):
        # Focal point distance fromn the origin of the spherical wave
        distance  *= 0.00001
        spherical  = beam.spherical(pxx,pxy,distance,wavelength,onepxtom,focal,1,'converging')
        aperture.show(diffrac.intensity(spherical,onepxtom),onepxtom,wavelength,'Detector at %s m' % distance)
    sys.exit()
    # Sample Fresnel and Fraunhofer region calculation of the given aperture
    for distance in xrange(1,10):
        distance    *= 0.01
        print 'lambda*d/w = %s m' % (wavelength*distance/(aperturesize*onepxtom))
        # Calculating far field behaviour
        output       = diffrac.fresnelfraunhofer(rectangle,wavelength,distance,onepxtom,aperturesize)
        # Calculating the fresnel number
        fresnelno    = diffrac.fresnelnumber(aperturesize,onepxtom,wavelength,distance)
        aperture.show(diffrac.intensity(output,onepxtom),onepxtom,wavelength,'Distance: %s m Wavelength: %s m Fresnel Number: %s'% (distance,wavelength,fresnelno))   
        aperture.showrow(diffrac.intensity(output,onepxtom),wavelength,onepxtom,distance)
    return True
    
if __name__ == '__main__':
    sys.exit(example())
