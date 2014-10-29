#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,odak
from math import *
from numpy import zeros
from pylab import *

__author__  = ('Kaan Akşit')

def LCD():
    greenwavelength = 532*pow(10,-9)
    nx              = 1
    ny              = 0.9
    d               = pow(10,-3)
    jones           = odak.jonescalculus()
    InputE          = jones.electricfield(1.,0.)
    rot1            = 40.; ret1 = 27.
    Outputs         = zeros((91,91))
    for rot1 in xrange(0,91):
        for ret1 in xrange(0,91,10):
#    rot1            = 45.; ret1 = 30.
            LCD1            = jones.waveplate(InputE,ret1,rot1)
#            LCD2            = jones.waveplate(InputE,ret1,rot1)
#    rot2            = 40.; ret2 = 27.
#    rot2            = 45.; ret2 = 30.
#    LCD2            = jones.waveplate(LCD1,ret2,rot2)
#    Output          = jones.linearpolarizer(LCD2,90)
            Output          = jones.linearpolarizer(LCD1,90)
            Intensity       = abs(Output[1])            
            Outputs[rot1,ret1] = Intensity
    plot(Outputs)
    ylabel('Intensity (Normalized)')
    xlabel('Angle between fast axis of the LCD and the input light polarization (degrees)')
    show()
#    print 'Input electric field vector: \n', InputE
#    print 'After the first LCD (Operational): \n', LCD1
#    print 'After the second LCD (Operational): \n', LCD2
#    print 'Ouput:', Output
#    print 'Intensity:', abs(Output[1])
    return True

if __name__ == '__main__':
    sys.exit(LCD())
