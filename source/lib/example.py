#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,odak

__author__  = ('Kaan Akşit')

def example():
    #example_of_gaussian()
    #example_of_spherical_wave()
    #example_of_fresnel_fraunhofer()
    example_of_retroreflector()
    return True

def example_of_retroreflector():
    onepxtom     = pow(10,-5)
    distance     = 1
    wavelength   = 500*pow(10,-9)
    aperturesize = 40
    pxx          = 240
    pxy          = 240
    pitch        = 40
    diffrac      = odak.diffractions()
    aperture     = odak.aperture()
    beam         = odak.beams()
    # Retroreflector corner cube array is created
    retro        = aperture.retroreflector(pxx,pxy,wavelength,pitch)
    aperture.show(retro,onepxtom,wavelength,'Detector')
    #aperture.show3d(retro)
    # Divergin gaussian beam defined
    focal        = 0
    amplitude    = 1
    waistsize    = 0.1
    gaussianbeam = beam.gaussian(pxx,pxy,distance,wavelength,onepxtom,amplitude,waistsize,focal)
    aperture.show(gaussianbeam,onepxtom,wavelength,'Detector at %s m' % (distance))
    # Output after the gaussian beam reflects from retroreflector
    output1      = gaussianbeam*retro
    aperture.show(output1,onepxtom,wavelength,'Detector')
    # Output at the far distance
    distance     = 0.8
    output2      = diffrac.fresnelfraunhofer(output1,wavelength,distance,onepxtom,aperturesize)
    aperture.show(diffrac.intensity(output2,onepxtom),onepxtom,wavelength,'Detector')
    return True

def example_of_gaussian():
    # Fixed values are set
    onepxtom     = pow(10,-5)
    distance     = 0.7
    wavelength   = 500*pow(10,-9)
    aperturesize = 40
    pxx          = 128
    pxy          = 128
    diffrac      = odak.diffractions()
    aperture     = odak.aperture()
    beam         = odak.beams()
    #aperture.show(rectangle,onepxtom,wavelength,'Aperture')
    # Defining a gaussian beam
    amplitude    = 10
    waistsize    = 20*onepxtom
    # Distance in between beam waist and the simulation origin
    focal        = 0.5
    for distance in xrange(1,3):
        distance    *= 0.2
        gaussianbeam = beam.gaussian(pxx,pxy,distance,wavelength,onepxtom,amplitude,waistsize,focal)
        aperture.show(diffrac.intensity(gaussianbeam,onepxtom),onepxtom,wavelength,'Detector at %s m' % (distance))
    aperture.show3d(gaussianbeam)
    return True

def example_of_spherical_wave():
    # Fixed values are set
    onepxtom     = pow(10,-5)
    distance     = 0.7
    wavelength   = 500*pow(10,-9)
    aperturesize = 40
    pxx          = 128
    pxy          = 128
    diffrac      = odak.diffractions()
    aperture     = odak.aperture()
    beam         = odak.beams()
    # Defining a diverging spherical wave
    focal        = 0.0001
    for distance in xrange(1,20):
        # Focal point distance fromn the origin of the spherical wave
        distance  *= 0.00001
        spherical  = beam.spherical(pxx,pxy,distance,wavelength,onepxtom,focal,1)
        aperture.show(diffrac.intensity(spherical,onepxtom),onepxtom,wavelength,'Detector at %s m' % distance)
    aperture.show3d(spherical)
    return True

def example_of_fresnel_fraunhofer():
    # Fixed values are set
    onepxtom     = pow(10,-5)
    distance     = 0.7
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
    aperture.show(rectangle,onepxtom,wavelength,'Aperture')
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
    aperture.show3d(diffrac.intensity(output,onepxtom))
    return True

if __name__ == '__main__':
    sys.exit(example())
