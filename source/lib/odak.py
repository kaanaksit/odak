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
        obj = zeros((nx,ny),dtype=complex)
        for i in xrange(nx):
            for j in xrange(ny):   
                obj[i,j] = 1/pi/pow(sigma,2)*exp(-float(pow(i-nx/2,2)+pow(j-ny/2,2))/2/pow(sigma,2))
        return obj
    def retroreflector(self,nx,ny,wavelength,pitch):
        if nx != ny:
           nx = max([nx,ny])
           ny = nx
        part  = zeros((pitch,int(pitch/2)))
        for i in xrange(int(sqrt(3)*pitch/6)):
            for j in xrange(int(pitch/2)):
                if float(j)/(int(sqrt(3)*pitch/6)-i) < sqrt(3): 
                    part[i,j]       = int(sqrt(3)*pitch/6)-i
                    part[pitch-i-1,int(pitch/2)-j-1] = part[i,j]
        for i in xrange(int(pitch)):
            for j in xrange(int(pitch/2)):
                if j != 0:
                    if float(j)/(int(pitch)-i) < 0.5 and (int(sqrt(3)*pitch/6)-i)/float(j) < 1./sqrt(3):
                        # Distance to plane determines the level of the amplitude 
                        # Plane as a line y = slope*x+ pitch 
                        # Perpendicula line  y = -(1/slope)*x+n
                        slope     = -0.5
                        n         = j + (1/slope) * (i)
                        x1        = (n - pitch/2)/(slope+1/slope)
                        y1        = -(1/slope)*x1+n 
                        part[i,j] = int(sqrt(3)*pitch/6) - sqrt( pow(i-x1,2) + pow(j-y1,2) )
                        part[pitch-i-1,int(pitch/2)-j-1] = part[i,j]
                else:
                    if i > int(sqrt(3)*pitch/6):
                        slope     = -0.5
                        n         = j + (1/slope) * (i)
                        x1        = (n - pitch/2)/(slope+1/slope)
                        y1        = -(1/slope)*x1+n
                        part[i,j] = int(sqrt(3)*pitch/6) - sqrt( pow(i-x1,2) + pow(j-y1,2) )
                        part[pitch-i-1,int(pitch/2)-j-1] = part[i,j]
        left  = part
        right = part[::-1]
        part  = append(left,right,axis=1)
        obj   = tile(part,(nx/pitch,ny/pitch))
        for i in xrange(nx/pitch/2):
           obj[(2*i+1)*pitch:(2*i+1)*pitch+pitch,:] = roll(obj[(2*i+1)*pitch:(2*i+1)*pitch+pitch,:],pitch/2)    
        return obj
    def show(self,obj,pixeltom,wavelength,title='Detector'):
        # Plots a detector showing the given object
        plt.figure(),plt.title(title)
        nx,ny = obj.shape
        # Number of the ticks to be shown both in x and y axes
        img = plt.imshow(abs(obj),cmap=matplotlib.cm.jet)
        plt.colorbar(img,orientation='horizontal')
        plt.show()
        return True
    def show3d(self,obj):
        nx,ny   = obj.shape
        fig     = plt.figure()
        ax      = fig.gca(projection='3d')
        X,Y     = mgrid[0:nx,0:ny]
        surf    = ax.plot_surface(X,Y,abs(obj), rstride=1, cstride=1, cmap=matplotlib.cm.jet,linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        return True
    def showrow(self,obj,wavelength,pixeltom,distance):
        # Plots row crosssection of the given object
        nx,ny = obj.shape
        a     = 5
        plt.figure()
        plt.plot(arange(-nx/2,nx/2)*pixeltom,abs(obj[nx/2,:]))
        plt.show()
        return True

class beams():
    def __init(self):
        return
    def spherical(self,nx,ny,distance,wavelength,pixeltom,focal,amplitude=1):
        # Spherical wave
        distance = abs(focal-distance)
        k        = 2*pi/wavelength
        X,Y      = mgrid[-nx/2:nx/2,-ny/2:ny/2]*pixeltom
        r        = sqrt(pow(X,2)+pow(Y,2)+pow(distance,2)) 
        U        = amplitude/r*exp(-1j*k*r)
        return U
    def gaussian(self,nx,ny,distance,wavelength,pixeltom,amplitude,waistsize,focal=0):
        # Gaussian beam
        distance = abs(distance-focal)
        X,Y      = mgrid[-nx/2:nx/2,-ny/2:ny/2]*pixeltom
        ro       = sqrt(pow(X,2)+pow(Y,2))
        z0       = pow(waistsize,2)*pi/wavelength
        A0       = amplitude/(1j*z0)
        if distance == 0:
            U = A0*exp(-pow(ro,2)/pow(waistsize,2))
            return U
        k        = 2*pi/wavelength
        R        = distance*(1+pow(z0/distance,2))
        W        = waistsize*sqrt(1+pow(distance/z0,2))
        ksi      = 1./arctan(distance/z0)
        U        = A0*waistsize/W*exp(-pow(ro,2)/pow(W,2))*exp(-1j*k*distance-1j*pow(ro,2)/2/R+1j*ksi)
        return U

class diffractions():
    def __init__(self):
        return
    def fresnelfraunhofer(self,wave,wavelength,distance,pixeltom,aperturesize):
        nu,nv  = wave.shape
        k      = 2*pi/wavelength
        X,Y    = mgrid[-nu/2:nu/2,-nv/2:nv/2]*pixeltom
        Z      = pow(X,2)+pow(Y,2)
        distancecritical = pow(aperturesize*pixeltom,2)*2/wavelength
        print 'Critical distance of the system is %s m. Distance of the detector is %s m.' % (distancecritical,distance)
        # Convolution kernel for free space
        h      = exp(1j*k*distance)/sqrt(1j*wavelength*distance)*exp(1j*k*0.5/distance*Z)
        qpf    = exp(1j*k*0.5/distance*Z)
        if distancecritical < distance:
            wave = wave*qpf
        result = fftshift(ifft2(fft2(wave)*fft2(h)))
        return result
    def fresnelnumber(self,aperturesize,pixeltom,wavelength,distance):
        return  pow(aperturesize*pixeltom,2)/wavelength/distance
    def intensity(self,obj,pixeltom):
        return abs(pow(obj,2))*pow(pixeltom,2)*0.5*8.854*pow(10,-12)*299792458

def ana():
    print 'Odak'    
    return True

if __name__ == '__main__':
    sys.exit(ana())
