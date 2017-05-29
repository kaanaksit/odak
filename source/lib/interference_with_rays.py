#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys,odak,math
import matplotlib.pyplot as plt
import scipy.misc

__author__  = ('Kaan Akşit')

def pattern(distance):
    ray      = odak.raytracing()
    source0    = np.array([ 1.,0.,0.])
    source1    = np.array([-1.,0.,0.])
    rectx      = 20.
    recty      = 20.
    nx         = 1000
    ny         = 1000
    wavelength = 532*10**-9
    result     = np.zeros((nx,ny))
    for ix in range(0,nx): 
        for iy in range(0,ny):
            x     = rectx*2./nx*ix-rectx
            y     = recty*2./ny*iy-recty
            point = np.array([x,y,distance])
            z0    = np.sqrt(np.sum((point-source0)**2))*2*np.pi/wavelength
            z1    = np.sqrt(np.sum((point-source1)**2))*2*np.pi/wavelength
            pd    = np.abs(z0-z1)
            result[ix,iy] = 1.+np.cos(pd)
            y=0
#    plt.imshow(result,cmap='gray')
#    plt.colorbar()
#    plt.show()
    scipy.misc.imsave('/output/result_%s.png' % distance,result)
    print(distance)
    return True

def main():
    for distance in np.linspace(1.,500.,25.):
        pattern(distance)
    return True

if __name__ == '__main__':
    sys.exit(main())
