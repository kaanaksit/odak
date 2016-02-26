#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,odak,math

__author__  = ('Kaan Akşit')

def example():
#    example_of_gaussian()
#    example_of_spherical_wave()
    #example_of_fresnel_fraunhofer()
    #example_of_retroreflector()
    #example_of_jones_calculus()
    #example_of_ray_tracing()
    #example_of_paraxial_matrix()
    example_of_ray_tracing_2()
    #example_of_ray_tracing_3()
    return True

def example_of_ray_tracing_3():
    ray         = odak.raytracing()
    # Be careful points and pitch must be float
    pitch       = 10.
    cornercube0 = ray.plotcornercube(0,0,0,pitch,revert=False)
    cornercube1 = ray.plotcornercube(pitch/math.sqrt(3),-pitch/3,0,pitch,revert=True)
    vector0     = ray.createvector((1.,2.1,10.),(90,90,0))
    vector1     = ray.createvector((1.,1.,10.),(90,90,0))
    vector2     = ray.createvector((4.,-4.,10.),(90,90,0))
    vectorlist  = [vector0,vector1,vector2]
    cornercubes = [cornercube0,cornercube1]
    for vectors in vectorlist:
        for cubes in cornercubes:
            rayslist          = []
            distance          = 2
            rayslist.append(vectors)
            for rays in rayslist:
                if len(rayslist) > 3:
#                    rays[1] = -rays[1]
                    ray.plotvector(rays,10)
                    break
                for points in cubes:
                    distance,normvec = ray.findintersurface(rays,(points[0],points[1],points[2]))
                    if ray.isitontriangle(normvec[0],points[0],points[1],points[2]) == True:
                        ray.plotvector(normvec,pitch/10,'r')
                        ray.plotvector(rays,distance)
                        reflectvector = ray.reflect(rays,normvec)
                        rayslist.append(reflectvector)
    ray.defineplotshape((-5,5),(-5,5),(0,10))
    ray.showplot()
    return True

def example_of_ray_tracing_2():
    ray               = odak.raytracing()
    point             = (0,math.sqrt(3)/2,1)
    point0            = (1,0,1)
    point1            = (0,1,0)
    point2            = (-1,0,1)
    ray.plottriangle(point0,point1,point2)
    vector            = ray.createvector((0,0.5,5),(90,90,0))
    distance,normvec  = ray.findintersurface(vector,(point0,point1,point2))
    if ray.isitontriangle(normvec[0],point0,point1,point2) == True:
        ray.plotvector(normvec,distance,'r')
        ray.plotvector(vector,distance,'g')
        reflectvector     = ray.reflect(vector,normvec)
        ray.plotvector(reflectvector,distance,'b')
    ray.defineplotshape((-20,20),(-20,20),(-20,20))
    ray.showplot()
    return True

def example_of_paraxial_matrix():
    # Call from odak library for paraxialmatrix calculations.   
    parax     = odak.ParaxialMatrix()
    # Start position of everything at X-axis.
    posx      = 0.
    # Input vector creation. 
    vector1   = parax.CreateVector(0,2) 
    print 'Input vector: \n%s' % vector1
    # Free-space propagation.
    distance  = 10.
    vector2   = parax.FreeSpace(vector1,distance)
    print 'Vector at a given certain distance: \n%s' % vector2
    # Plotting the ray.
    posx      = parax.PlotVector(vector1,vector2,posx)
    # Show plot.
    parax.ShowPlot()
    return True

def example_of_ray_tracing():
    n         = 1
    ray       = odak.raytracing()
#    rotvec = ray.transform(vector,(45,0,0),(0.5,1,0))
#    print 'Output vector: \n %s' % rotvec
    spherical = ray.plotsphericallens(20,0,0,10)
    for angle in xrange(100,120):
        vector            = ray.createvector((0,5,5),(45,angle,angle))
        distance,normvec  = ray.findinterspher(vector,spherical)
        if distance != 0:      
            ray.plotvector(vector,distance)
            refractvector = ray.snell(vector,normvec,1.0,1.51)
            #ray.plotvector(refractvector,20)
            #reflectvector     = ray.reflect(vector,normvec)
            #ray.plotvector(reflectvector,distance)
        distance,normvec  = ray.findinterspher(refractvector,spherical)
        if distance != 0:
            ray.plotvector(refractvector,distance)
            refractvector2 = ray.snell(refractvector,normvec,1.51,1.)
            ray.plotvector(refractvector2,20)
    ray.showplot()
    return True

def example_of_jones_calculus():
    greenwavelength = 532*pow(10,-9)
    redwavelength   = 432*pow(10,-9)
    bluewavelength  = 640*pow(10,-9)
    nx              = 1
    ny              = 0.9
    d               = pow(10,-3)
    jones           = odak.jonescalculus()
    print 'A sample linear polarizer: \n', jones.linearpolarizer(1,90)
    print 'A sample circullar polarizer: \n', jones.circullarpolarizer(1,'lefthanded')
    print 'A sample quarter wave plate: \n', jones.quarterwaveplate(1,0)
    print 'A sample half wave plate: \n', jones.halfwaveplate(1,0)
    print 'A sample birefringent plate: \n', jones.birefringentplate(1,nx,ny,d,greenwavelength,0)
    print 'A sample nematic liquid crystal cell: \n', jones.nematicliquidcrystal(1,3000,1.2,1,0.1,greenwavelength,0)
    print 'A sample ferroelectric liquid crystal cell: \n', jones.ferroliquidcrystal(1,30,2,1,0.1,greenwavelength,'+',0) 
    return True

def example_of_retroreflector():
    onepxtom     = pow(10,-5)
    distance     = 0.8
    wavelength   = 500*pow(10,-9)
    aperturesize = 40
    pxx          = 1920
    pxy          = 1920
    pitch        = 40
    diffrac      = odak.diffractions()
    aperture     = odak.aperture()
    beam         = odak.beams()
    # Retroreflector corner cube array is created
    retro        = aperture.retroreflector(pxx,pxy,wavelength,pitch,'normal')
    aperture.show(retro,onepxtom,wavelength,'Detector')
    #aperture.show(diffrac.fft(retro),onepxtom,wavelength,'FFT')
    #aperture.show3d(retro)
    # Divergin gaussian beam defined
    focal        = 0.8
    amplitude    = 1
    waistsize    = 50*onepxtom
    gaussianbeam = beam.gaussian(pxx,pxy,distance,wavelength,onepxtom,amplitude,waistsize,focal)
    aperture.show(gaussianbeam,onepxtom,wavelength,'Detector at %s m' % (distance))
    # Output after the gaussian beam reflects from retroreflector
    output1      = gaussianbeam*retro
    aperture.show(output1,onepxtom,wavelength,'Detector')
    # Output at the far distance
    distance     = 1
    output2      = diffrac.fresnelfraunhofer(output1,wavelength,distance,onepxtom,aperturesize)
    aperture.show(diffrac.intensity(output2,onepxtom),onepxtom,wavelength,'Detector','normal')
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
    focal        = 8
    for distance in xrange(5,10):
        distance    *= 1
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
    # Fixed values are set.
    onepxtom     = pow(10,-6)
    # Aperture size in mm.
    wavelength   = 500*pow(10,-9)
    pxx          = 2048
    pxy          = pxx
    diffrac      = odak.diffractions()
    aperture     = odak.aperture()
    beam         = odak.beams()
    # A spherical beam traveled from a point to a pinhole, with a distance in mm.
    focal        = pow(10,-4)
    distance     = 9.
    distance    *= pow(10,-3)
    spherical    = beam.spherical(pxx,pxy,distance,wavelength,onepxtom,focal,1)
#    aperture.show(spherical,onepxtom,wavelength,'Spherical wave')
    for i in xrange(1,20):
        da           = i*0.01
        aperturesize = da*pow(10,-3)/onepxtom
        circle       = aperture.circle(pxx,pxy,aperturesize)
    #    aperture.show(circle,onepxtom,wavelength,'Aperture')
        # Spherical wave transmitted from a pinhole.
        AfterPinhole = diffrac.transmittance(spherical,circle)
    #    aperture.show(AfterPinhole,onepxtom,wavelength,'After a pinhole')
        # Distance between pinhole and a lens in mm.
        distance     = 43.
        distance    *= pow(10,-3)
        # Sample Fresnel and Fraunhofer region calculation of the given aperture
        print 'lambda*d/w = %s m' % (wavelength*distance/(aperturesize*onepxtom))
        # Calculating far field behaviour.
        BeforeLens   = diffrac.fresnelfraunhofer(AfterPinhole,wavelength,distance,onepxtom,aperturesize)
        # Calculating the fresnel number.
        fresnelno    = diffrac.fresnelnumber(aperturesize,onepxtom,wavelength,distance)
    #    aperture.show(diffrac.intensity(BeforeLens,onepxtom),onepxtom,wavelength,'Before a lens, Distance: %s m Wavelength: %s m Fresnel Number: %s'% (distance,wavelength,fresnelno))   
    #    aperture.showrow(diffrac.intensity(BeforeLens,onepxtom),wavelength,onepxtom,distance)
        # Multiply it with lens transmittance function. Focal length in mm.
        focal        = 22. 
        focal       *= pow(10,-3)
        AfterLens    = diffrac.lens(BeforeLens,wavelength,focal,onepxtom)
        # Calculating far field behaviour.
        distance     = 22.
        distance    *= pow(10,-3)
        aperturesize = 20.*pow(10,-3)/onepxtom
        output       = diffrac.fresnelfraunhofer(AfterLens,wavelength,distance,onepxtom,aperturesize)
        # Calculating the fresnel number.
        fresnelno    = diffrac.fresnelnumber(aperturesize,onepxtom,wavelength,distance)
        aperture.show(diffrac.intensity(output,onepxtom),onepxtom,wavelength,'Retina, Distance: %s m Wavelength: %s m Fresnel Number: %s'% (distance,wavelength,fresnelno),'normal','da=%smm.png'%da)   
        aperture.showrow(diffrac.intensity(output,onepxtom),wavelength,onepxtom,distance,'row_da=%smm.png' % da)    
        # Show a 3D plot of the output.
    #    aperture.show3d(diffrac.intensity(output,onepxtom))
    # Showing plots.
#    aperture.showplots()
    return True

if __name__ == '__main__':
    sys.exit(example())
