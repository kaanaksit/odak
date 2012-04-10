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
    # Defining the aperture
    aperture     = odak.aperture()
    rectangle    = aperture.rectangle(pxx,pxy,aperturesize)
    gaussian     = aperture.gaussian(pxx,pxy,aperturesize)
    circle       = aperture.circle(pxx,pxy,aperturesize)
    aperture.show(rectangle,onepxtom,wavelength,distance,'Aperture')
#    aperture.show3d(circle)
    for distance in xrange(1,10):
        distance    *= 0.01
        print 'lambda*d/w = %s' % (wavelength*distance/(aperturesize*onepxtom))
        # Calculating far field behaviour
        diffrac      = odak.diffractions()
        output       = diffrac.fresnelfraunhofer(rectangle,wavelength,distance,onepxtom,aperturesize)
        # Calculating the fresnel number
        fresnelno    = diffrac.fresnelnumber(aperturesize,onepxtom,wavelength,distance)
        aperture.show(diffrac.intensity(output,onepxtom),onepxtom,wavelength,distance,'Distance: %s m Wavelength: %s m Fresnel Number: %s'% (distance,wavelength,fresnelno))   
        aperture.showrow(diffrac.intensity(output,onepxtom),wavelength,onepxtom,distance)
    return True
    
if __name__ == '__main__':
    sys.exit(example())
