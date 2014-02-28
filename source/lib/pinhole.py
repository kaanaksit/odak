#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,odak,math,csv
import matplotlib.pyplot as plt
from math import cos, sin

__author__  = ('Kaan Akşit')
# This script solves the question how stereoscopy is improved with two pinholes infront of an eye.

# Main function where variables are set, and the solution method is called.
def main():
    # REMEMBER TO ALWAYS ASSIGN FLOATING NUMBERS TO THE VARIABLES!
    # Distance between pinholes and the point sources (mm).
    ds          = 1000.0
    # Distance between center to center in between pinholes (mm).
    dhp         = 2.
    # Radius of the pinholes (mm).
    r           = 0.5
    # Distance between the point sources (mm).
    dpp         = 4.0
    # Half of the aperture of the eye (mm), and also determines equivalent ball lens focal.
    dea         = 9.0
    # Diameter of the whole eye, according to http://www.opticampus.com/files/introduction_to_ophthalmic_optics.pdf ,which employs Gullstrand-Emsley model.
    dwe         = 12.0
    # Half of the thickness of the eye lens (mm).
    tel         = 0.2
    # Refractive index of the cristal eye lens (ball lens in our case).
    nel         = 1.41
    # Refractive index of inside the lens.
    nie         = 1.33
    # Refractive index of the outside envorinment.
    nair        = 1.
    # Distance between the pinholes and the lens (mm).
    dpl         = -10.
    # X and Y positions of the lens.
    xel         = 0
    yel         = 0
    # Detector position in 3D space determined by three points from the surface. 
    DetPos      = [
                  (-30,30,-50),
                  (30,-30,-50),
                  (30,30,-50)
                  ]
    # Solve the problem according to the given values.
#    Solve(ds,dhp,r,dpp,dea,dwe,tel,nel,nie,nair,dpl,xel,yel,DetPos,True)
    # Open a csv file to store the output of the simulation.
    filename = 'results.csv'
    out      = csv.writer(open(filename,"w"), delimiter=',',quoting=csv.QUOTE_ALL)
    # Iterative solution for plotting dependency on aperture size, separation between point sources, and separation between two pinholes.
    VoxelHeights = []
    VoxelWidths  = []
    X            = []
    Y            = []
    Maxr         = 0.91
    Maxdpp       = 5.0
    step         = 0.1
    # Iterate the aperture size.
    for r in xrange(1,int(Maxr/step),1):
        r *= step
        # Iterate the seperation of point sources.
        for dpp in xrange(1,int(Maxdpp/step),1):
            dpp *= step
            # Solve the given case.
            VoxelHeight, VoxelWidth = Solve(ds,dhp,r,dpp,dea,dwe,tel,nel,nie,nair,dpl,xel,yel,DetPos,False)
            # Voxel dimensions stored in a list.
            VoxelHeights.append(VoxelHeight)
            VoxelWidths.append(VoxelWidth)
            X.append(r*2)
            Y.append(dpp)
            # Values stored in a list.
            Values = (r,dpp,VoxelHeight,VoxelWidth)
            # Show current the latest solution.
            print Values
            # Write the solution to the file.
            out.writerow(Values)
 
    # Font size in the plots.
    FontSize =  22
    # Font weight in the plots.
    FontWeight = 'normal'
    # Font type in the plots.
    FontType = 'sans'
    
    # Call ray tracing library to plot the data as a 3D surface.
    Fig = odak.raytracing()
    # Setting font properties in the figure.
    Fig.SetPlotFontSize(FontType,FontWeight,FontSize)
    
 
    # Call ray tracing library to plot the data as a 3D surface.
    Fig1 = odak.raytracing()
    # Setting font properties in the figure.
    Fig1.SetPlotFontSize(FontType,FontWeight,FontSize)
    # Plot the data as 3D surface.
    Fig1.PlotData(X,Y,VoxelHeights,'g')
    # Show the plotted data.
    Fig1.showplot('$h$ (mm)','$d_p$ (mm)', '$d_c$ (mm)', 'VoxelHeightRay.png')


    # Call ray tracing library to plot the data as a 3D surface.
    Fig2 = odak.raytracing()
    # Setting font properties in the figure.
    Fig2.SetPlotFontSize(FontType,FontWeight,FontSize)
    # Plot the data as 3D surface.
    Fig2.PlotData(X,Y,VoxelWidths,'g')
    # Show the plotted data.
    Fig2.showplot('$w$ (mm)','$d_p$ (mm)', '$d_c$ (mm)', 'VoxelWidthRay.png')
    return True

def Solve(ds,dhp,r,dpp,dea,dwe,tel,nel,nie,nair,dpl,xel,yel,DetPos,ShowPlot=False):
    # Z position of the lens is calculated.
    zel         = dpl-tel
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
    # Exterior points on the pinholes.
    PointList1 = [HoleCenter1] 
    PointList2 = [HoleCenter2] 
    SourceList = [Point1,
                  Point2
                 ]
    # Add additional points on the exterior of the pinholes.
    for k in xrange(0,360,60):
        # Converting degrees to radians.
        k = ray.DegreesToRadians(k)
        # Appending new item to the point lists.
        PointList1.append((HoleCenter1[0]+Radius1*cos(k),HoleCenter1[1]+Radius1*sin(k),HoleCenter1[2]))
        PointList2.append((HoleCenter2[0]+Radius2*cos(k),HoleCenter2[1]+Radius2*sin(k),HoleCenter2[2]))
    # Plot a spherical lens.
#    AsphericalLens = ray.plotasphericallens(xel,yel,zel,dea,dea,tel,nel,'b')
    # Create the whole eye ball.
    EyeBall  = (xel,yel,zel-dwe,dwe)
    # Plot the eye ball.
    SphericalLens = ray.plotsphericallens(EyeBall[0],EyeBall[1],EyeBall[2],EyeBall[3],'y',0.1)
    # Create a dummy spherical lens to find the intersection point of the incoming ray with aspherical lens.
    DummySL1 = (xel,yel,zel-dea,dea)
    # Plot a spherical lens.
    SphericalLens = ray.plotsphericallens(DummySL1[0],DummySL1[1],DummySL1[2],DummySL1[3],'g',0.1)
    # Calculate and print power of the spherical lens.
    D = (nel-nair)*(pow(dea,-1)-pow(-dea,-1)+(nel-nair)*2*dea/nel/(-pow(dea,2)))
#    print 'Power of the spherical lens: ', 1000.*D
#    print 'Focal length of the spherical lens (mm): ', 1./D     
    # Calculate the effective focal length of a ball lens.
    # See http://www.edmundoptics.com/technical-resources-center/optics/understanding-ball-lenses/
    EFL = nel*2*dea/4/(nel-nair)
#    print 'Effective focal length of the ball lens (mm):', EFL
    # Arrays to save rays by their source of origin.
    rays1    = []
    rays2    = []
    # Plot the pinholes.
    ray.PlotCircle(HoleCenter1,Radius1)
    ray.PlotCircle(HoleCenter2,Radius2)
    # Iterate ray tracing for every point source.
    for origin in SourceList:
        # Make destination choice according to the source, color the rays accordingly.
        if origin == Point1:
            DestinationList = PointList1
            RayColor           = 'r'
        elif origin == Point2:
            DestinationList = PointList2
            RayColor           = 'b'
        # Iterate ray tracing for every destination.
        for item in DestinationList:
            # Finding the angle in between the origin and the destination point.
            angles = ray.findangles(origin,item)
            # Creating the new vector in the direction of destination using the previous angles calculated.
            NewRay = ray.createvector(origin,angles)
            # Chiefray color selection.
            if item == DestinationList[0]:
                color = RayColor + '+--'
            else:
                color = RayColor + '+-'
            # Intersect the vector with dummy sphere.
            distance,normvec = ray.findinterspher(NewRay,DummySL1)  
            # Plot the ray from origin to the destination.
            if distance != 0:
                ray.plotvector(NewRay,distance,color)        
                # Storing ray to the rays vector by their source of origin.
                if origin == Point1:
                    rays1.append(NewRay)
                else:
                    rays2.append(NewRay)
                # Refracting into the eye.
                RefractRay = ray.snell(NewRay,normvec,nair,nel)
                # The ray travels inside the eye ball.
                distance,normvec = ray.findinterspher(RefractRay,DummySL1)
                # Plot the refracting ray.
                ray.plotvector(RefractRay,distance,color)
                # Refracting to outside of the eye.
                RefractOutsideRay = ray.snell(RefractRay,normvec,nel,nie)
                # Shine rays to the retina.
                distance,normvec = ray.findinterspher(RefractOutsideRay,EyeBall)
                # Find the intersection of the refracted ray with the detector surface.
#                distance,normvec = ray.findintersurface(RefractOutsideRay,(DetPos[0],DetPos[1],DetPos[2]))
#                # Plot the refracting ray.
                ray.plotvector(RefractOutsideRay,distance,color)
    # Loop to find intersection of the plotted rays.
    InterPoints = []
    for RaySource1 in rays1:
        for RaySource2 in rays2:
            # Find the intersection of two rays.
            intersection, distances = ray.CalculateIntersectionOfTwoVectors(RaySource1,RaySource2)
            # Check if the calculated distance value has a logical value.
            CheckValue = "%.2f" % abs(distances[0])
            if CheckValue != '0.00' and float(CheckValue) < 100000:
                ray.PlotPoint(intersection,'go',False,True)
                # Storing intersection point in a list.
                InterPoints.append(intersection)
    # Transpose of the InterPoints 2D list.
    l  = map(list,map(None,*InterPoints))
    # Finding Voxel height.
    m1          = max(l[2])
    m2          = min(l[2])
    VoxelHeight = abs(m1-m2) 
    # Finding Voxel width.
    m3         = max(l[0])
    m4         = min(l[0])
    VoxelWidth = abs(m3-m4)
    # Show the ray tracing envorinment in three-dimensional space.
    if ShowPlot == True:
        limit = 0.8*ds
        ray.defineplotshape((-limit,limit),(-limit,limit),(-limit,limit))
        ray.showplot()
    else:
        # Otherwise destroy figure.
        ray.CloseFigure()
    return VoxelHeight,VoxelWidth

if __name__ == '__main__':
    sys.exit(main())
