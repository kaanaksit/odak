#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,odak,math

__author__  = ('Kaan Akşit')

def LCD():
    greenwavelength = 532*pow(10,-9)
    nx              = 1
    ny              = 0.9
    d               = pow(10,-3)
    jones           = odak.jonescalculus()
    InputE          = jones.electricfield(1.,0.)
    rot1            = 45.
    LCD1            = jones.halfwaveplate(InputE,rot1)
    rot2            = 45.
    LCD2            = jones.halfwaveplate(LCD1,rot2)
    print 'Input electric field vector: \n', InputE
    print 'After the first LCD: \n', LCD1
    print 'After the second LCD: \n', LCD2
    return True

if __name__ == '__main__':
    sys.exit(LCD())
