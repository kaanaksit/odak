#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,odak,math

__author__  = ('Kaan Akşit')

def example():
    PinholeCircular()
    return True

def PinholeCircular():
    # REMEMBER TO ALWAYS ASSIGN FLOATING NUMBERS TO THE VARIABLES!
    # Distance between pinholes and the point sources (mm).
    ds          = 5.0
    # Distance between center to center in between pinholes (mm).
    dhp          = 1.5
    # Radius of the pinholes (mm).
    r           = 0.5
    # Distance between the point sources (mm).
    dpp         = 1.0
    # Define the center of the first circular pinhole in 3D space.
    HoleCenter1 = (-dhp/2,0,0)
    # Define the radius of the first circular pinhole in mm.
    Radius1     = r
    # Define the center of the second circular pinhole in 3D space.
    HoleCenter2 = (dhp/2,0,0)
    # Define the radius of the second circular pinhole in mm.
    Radius2     = r
    # Call the library to trace.
    ray         = odak.raytracing()
    # Position of the first point source.
    Point1      = (dpp/2,0,ds)
    # Position of the second point source.
    Point2      = (-dpp/2,0,ds)
    # First section of the solution for first point source and first pinhole.
    PointList1 = [HoleCenter1,
                 (HoleCenter1[0]+Radius1,HoleCenter1[1],HoleCenter1[2]),
                 (HoleCenter1[0]-Radius1,HoleCenter1[1],HoleCenter1[2]),
                 (HoleCenter1[0],HoleCenter1[1]+Radius1,HoleCenter1[2]),
                 (HoleCenter1[0],HoleCenter1[1]-Radius1,HoleCenter1[2])
                ] 
    PointList2 = [HoleCenter2,
                 (HoleCenter2[0]+Radius2,HoleCenter2[1],HoleCenter2[2]),
                 (HoleCenter2[0]-Radius2,HoleCenter2[1],HoleCenter2[2]),
                 (HoleCenter2[0],HoleCenter2[1]+Radius2,HoleCenter2[2]),
                 (HoleCenter2[0],HoleCenter2[1]-Radius2,HoleCenter2[2])
                ] 
    SourceList = [Point1,
                  Point2
                 ]
    # Iterate ray tracing for every point source.
    for origin in SourceList:
        # Make destination choice according to the source.
        if origin == Point1:
            DestinationList = PointList1
            RayColor           = 'r'
            ray.plotcircle(HoleCenter1,Radius1)
        elif origin == Point2:
            DestinationList = PointList2
            RayColor           = 'b'
            ray.plotcircle(HoleCenter2,Radius2)
        # Iterate ray tracing for every destination.
        for item in DestinationList:
            # Finding the angle in between the origin and the destination point.
            angles = ray.findangles(origin,item)
            # Creating the new vector in the direction of destination using the previous angles calculated.
            NewRay = ray.createvector(origin,angles)
            # Find the distance between the origin and the destination point.
            distance    = ray.finddistancebetweentwopoints(origin,item)
            # Chiefray color selection.
            if item == DestinationList[0]:
                color = RayColor + '+--'
            else:
                color = RayColor + '+-'
            # Plot the ray from origin to the destination.
            ray.plotvector(NewRay,distance,color)        
    # Show the ray tracing envorinment in three-dimensional space.
    ds = 1.5*ds
    ray.defineplotshape((-ds,ds),(-ds,ds),(-ds,ds))
    ray.showplot()
    return True

if __name__ == '__main__':
    sys.exit(example())
