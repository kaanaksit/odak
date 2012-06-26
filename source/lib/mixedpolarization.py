#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,odak,numpy

__author__  = ('Kaan Akşit')


def main():
    # Initial variables
    greenwavelength = 532*pow(10,-9)
    redwavelength   = 432*pow(10,-9)
    bluewavelength  = 640*pow(10,-9)
    nx              = 2
    ny              = 1
    ni              = 360
    nj              = 360
    ratio           = numpy.zeros((ni,nj))
    crosstalk       = numpy.zeros((ni,nj))
    jones           = odak.jonescalculus()
    aperture        = odak.aperture()
    # Initial electric field vector defined
    u               = jones.electricfield(1/pow(2,0.5),1/pow(2,0.5))
    # Apply rotation combination
    for wavelength in [redwavelength,greenwavelength,bluewavelength]:
        for i in xrange(0,ni):
            for j in xrange(0,nj):
                # Solve the system
                urot           = jones.birefringentplate(u,nx,ny,greenwavelength/2,wavelength,i)
                uqwp           = jones.birefringentplate(urot,nx,ny,greenwavelength/4,wavelength,j)
                ul             = jones.circullarpolarizer(uqwp,'lefthanded')
                ur             = jones.circullarpolarizer(uqwp,'righthanded')
                # Calculate crosstalk and illumination ratio on left and right
                crosstalk[i,j] = (abs(ur[0])+abs(ur[1]))/(abs(u[0])+abs(u[1]))
                ratio[i,j]     = (abs(ur[0])+abs(ur[1]))/(abs(ul[0])+abs(ul[1]))
        # Show crosstalk and illumination ratio on left and right
        aperture.show(crosstalk,1,wavelength,'Crosstalk at wavelength %s' % wavelength,'normal',filename='Crosstalk%s.png' % wavelength,xlabel='Rotation of QWP (degrees)', ylabel='Rotation of polarization rotator (degrees)')
        aperture.show(ratio,1,wavelength, 'Ratios of the eyes at wavelength %s' % wavelength,'normal',filename='Ratio%s.png' % wavelength,xlabel='Rotation of QWP (degrees)',ylabel='Rotation of polarization rotator (degrees)')
    return True

if __name__ == '__main__':
    sys.exit(main())
