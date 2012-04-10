#!/usr/bin/python
# -*- coding: utf-8 -*-
# Programmed by Kaan Akşit

import sys,matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from numpy import *
from numpy.fft import *

__author__  = ('Kaan Akşit')

class aperture():
    def __init__(self):
        return
    def twoslits(self,nx,ny,X,Y,delta):
        # Creates a matrix that contains two slits
        obj=zeros((nx,ny),dtype=complex)
        for i in range(int(nx/2-X/2),int(nx/2+X/2)):
            for j in range(int(ny/2+delta/2-Y/2),int(ny/2+delta/2+Y/2)):
                obj[ny/2-abs(ny/2-j),i] = 1
                obj[j,i] = 1      
        return obj
    def rectangle(self,nx,ny,side):
        # Creates a matrix that contains rectangle
        obj=zeros((nx,ny),dtype=complex)
        for i in range(int(nx/2-side/2),int(nx/2+side/2)):
            for j in range(int(ny/2-side/2),int(ny/2+side/2)):
                obj[j,i] = 1 
        return obj
    def circle(self,nx,ny,radius):
        # Creates a matrix that contains circle
        obj=zeros((nx,ny),dtype=complex)
        for i in range(int(nx/2-radius/2),int(nx/2+radius/2)):
            for j in range(int(ny/2-radius/2),int(ny/2+radius/2)):
                if (abs(i-nx/2)**2+abs(j-ny/2)**2)**(0.5)< radius/2:
                    obj[j,i] = 1 
        return obj
    def sinamgrating(self,nx,ny,grating):
        # Creates a sinuosidal grating matrix
        obj=zeros((nx,ny),dtype=complex)
        for i in xrange(nx):
            for j in xrange(ny):
                obj[i,j] = 0.5+0.5*cos(2*pi*j/grating)
        return obj
    def lens(self,nx,ny,focal,wavelength):
        # Creates a lens matrix
        obj = zeros((nx,ny),dtype=complex)
        k   = 2*pi/wavelength
        for i in xrange(nx):
            for j in xrange(ny):
                obj[i,j] = exp(-1j*k*(pow(i,2)+pow(j,2))/2/focal)    
        return obj
    def gaussian(self,nx,ny,sigma):
        # Creates a 2D gaussian matrix
        obj=zeros((nx,ny),dtype=complex)
        for i in xrange(nx):
            for j in xrange(ny):   
                obj[i,j] = 1/pi/pow(sigma,2)*exp(-float(pow(i-nx/2,2)+pow(j-ny/2,2))/2/pow(sigma,2))
        return obj
    def show(self,obj,pixeltom,wavelength,distance,title='Detector'):
        # Plots a detector showing the given object
        plt.figure(),plt.title(title)
        nx,ny = obj.shape
        # Number of the ticks to be shown both in x and y axes
        a     = 5
#        plt.xticks(nx/a*(arange(a)+1),nx/a*(arange(a)+1)*pixeltom)
#        plt.yticks(ny/a*(arange(a)+1),ny/a*(arange(a)+1)*pixeltom)
        plt.imshow(abs(obj),cmap=matplotlib.cm.jet)
        plt.show()
        return True
    def show3d(self,obj):
        nx,ny   = obj.shape
        fig     = plt.figure()
        ax      = fig.gca(projection='3d')
        X,Y     = meshgrid(arange(nx),arange(ny))
        surf = ax.plot_surface(X, Y, abs(obj), rstride=1, cstride=1, cmap=matplotlib.cm.jet,linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        return True
    def showrow(self,obj,wavelength,pixeltom,distance):
        # Plots row crosssection of the given object
        nx,ny = obj.shape
        a     = 5
        plt.figure()
#        plt.xticks(nx/a*(arange(a)+1),nx/a*(arange(a)+1)*pixeltom/wavelength/distance)
        plt.plot(arange(-nx/2,nx/2)*pixeltom,abs(obj[nx/2,:]))
        plt.show()
        return True

class diffractions():
    def __init__(selfparams):
        return
    def fresnelfraunhofer(self,wave,wavelength,distance,pixeltom,aperturesize):
        nu,nv  = wave.shape
        k      = 2*pi/wavelength
        X,Y    = mgrid[-nu/2:nu/2,-nv/2:nv/2]*pixeltom
        Z      = pow(X,2)+pow(Y,2)
        distancecritical = pow(aperturesize*pixeltom,2)*2/wavelength
        print 'Critical distance of the system is %s m. Distance of the detector is %s m.' % (distancecritical,distance)
        # Convolution kernel for Fresnel, Fourier multiplier for Fraunhofer
        h      = exp(1j*k*distance)/sqrt(1j*wavelength*distance)*exp(1j*k*0.5/distance*Z)
        qpf    = exp(-1j*k*0.5/distance*Z)
        if distancecritical < distance:
            wave = wave*qpf
        result = fftshift(ifft2(fft2(wave)*fft2(h)))
        return result
    def fresnelnumber(self,aperturesize,pixeltom,wavelength,distance):
        fresnelno = pow(aperturesize*pixeltom,2)/wavelength/distance
        return fresnelno
    def intensity(self,obj,pixeltom):
        return abs(pow(obj,2))*pow(pixeltom,2)*0.5*8.854*pow(10,-12)*299792458

def ana():
    print 'Odak'    
    return True

if __name__ == '__main__':
    sys.exit(ana())
