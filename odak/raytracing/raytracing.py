#!/usr/bin/env python
# coding: utf-8

# Standard libraries.
import os,sys,math
import numpy as np

# Other libraries.
sys.path.append('./')

# Globals.
global __author__
global __title__

__author__  = ('Kaan Akşit')
__title__   = 'Odak raytracing'
# See "General Ray tracing procedure" from G.H. Spencerand M.V.R.K Murty for the theoratical explanation.

def create_ray(x0y0z0,abg):
    """
    Definition to create a ray.

    Parameters
    ----------
    x0y0z0       : list
                   List that contains X,Y and Z start locations of a ray.
    abg          : list
                   List that contaings angles in degrees with respect to the X,Y and Z axes.

    Returns
    ----------
    ray          : ndarray
                   Array that contains starting points and cosines of a created ray.
    """
    # Due to Python 2 -> Python 3.
    x0,y0,z0         = x0y0z0
    alpha,beta,gamma = abg
    # Create a vector with the given points and angles in each direction
    point            = np.array([x0,y0,z0],dtype=np.float)
    alpha            = math.cos(np.radians(alpha))
    beta             = math.cos(np.radians(beta))
    gamma            = math.cos(np.radians(gamma))
    # Cosines vector.
    cosines          = np.array([alpha,beta,gamma],dtype=np.float)
    ray              = np.array([point,cosines],dtype=np.float)
    return ray

def create_ray_from_two_points(x0y0z0,x1y1z1):
    """
    Definition to create a ray from two given points.

    Parameters
    ----------
    x0y0z0       : list
                   List that contains X,Y and Z start locations of a ray.
    x1y1z1       : list
                   List that contains X,Y and Z ending locations of a ray.

    Returns
    ----------
    ray          : ndarray
                   Array that contains starting points and cosines of a created ray.
    s            : float
                   Total optical path differences in between start and end points of a created ray.
    """
    # Because of Python 2 -> Python 3.
    x0,y0,z0  = x0y0z0
    x1,y1,z1  = x1y1z1
    x0 = float(x0); y0 = float(y0); z0 = float(z0)
    x1 = float(x1); y1 = float(y1); z1 = float(z1)
    # Create a vector from two given points.
    point     = np.array([x0,y0,z0],dtype=np.float)
    # Distance between two points.
    s         = math.sqrt( (x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2 )
    if s != 0:
        alpha = (x1-x0)/s
        beta  = (y1-y0)/s
        gamma = (z1-z0)/s
    elif s == 0:
        alpha = float('nan')
        beta  = float('nan')
        gamma = float('nan')
    # Cosines vector
    cosines   = np.array([alpha,beta,gamma],dtype=np.float)
    # Returns vector and the distance.
    ray       = np.array([point,cosines],dtype=np.float)
    return ray,s

def multiply_two_vectors(vector1,vector2):
    """
    Definition to Multiply two vectors and return the resultant vector. Used method described under: http://en.wikipedia.org/wiki/Cross_product

    Parameters
    ----------
    vector1      : ndarray
                   A vector/ray.
    vector2      : ndarray
                   A vector/ray.

    Returns
    ----------
    ray          : ndarray
                   Array that contains starting points and cosines of a created ray.
    """
    angle  = np.cross(vector1[1],vector2[1])
    angle  = np.asarray(angle)
    ray    = np.array([vector1[0],angle],dtype=np.float)
    return ray

def find_intersection_w_surface(ray,points):
    """
    Definition to find intersection point inbetween a surface and a ray. For more see: http://geomalgorithms.com/a06-_intersect-2.html

    Parameters
    ----------
    vector       : ndarray
                   A vector/ray.
    points       : ndarray
                   Set of points in X,Y and Z to define a planar surface.

    Returns
    ----------
    normal       : ndarray
                   Surface normal at the point of intersection.
    distance     : float
                   Distance in between starting point of a ray with it's intersection with a planar surface.
    """
    point0,point1,point2 = points
    vector0,s            = create_ray_from_two_points(point0,point1)
    vector1,s            = create_ray_from_two_points(point1,point2)
    vector2,s            = create_ray_from_two_points(point0,point2)
    normal               = multiply_two_vectors(vector0,vector2)
    f                    = point0-ray[0]
    n                    = normal[1].copy()
    distance             = np.dot(n,f)/np.dot(n,ray[1])
    normal[0][0]         = ray[0][0]+distance*ray[1][0]
    normal[0][1]         = ray[0][1]+distance*ray[1][1]
    normal[0][2]         = ray[0][2]+distance*ray[1][2]
    return normal,distance

def reflect(input_ray,normal):
    """
    Definition to reflect an incoming ray from a surface defined by a surface normal. Used method described in G.H. Spencer and M.V.R.K. Murty, "General Ray-Tracing Procedure", 1961.

      Parameters
      ----------
      input_ray    : ndarray
                     A vector/ray.
      normal       : ndarray
                     A surface normal.

      Returns
      ----------
      output_ray   : ndarray
                     Array that contains starting points and cosines of a reflected ray.
      """
      mu               = 1
      div              = pow(normal[1,0],2)  + pow(normal[1,1],2) + pow(normal[1,2],2)
      a                = mu* ( input_ray[1][0]*normal[1][0]
                             + input_ray[1][1]*normal[1][1]
                             + input_ray[1][2]*normal[1][2]) / div
      output_ray       = input_ray.copy()
      output_ray[0][0] = normal[0][0]
      output_ray[0][1] = normal[0][1]
      output_ray[0][2] = normal[0][2]
      output_ray[1][0] = input_ray[1][0] - 2*a*normal[1][0]
      output_ray[1][1] = input_ray[1][1] - 2*a*normal[1][1]
      output_ray[1][2] = input_ray[1][2] - 2*a*normal[1][2]
      return output_ray

def define_plane(point,angles=[0.,0.,0.]):
    """
    Definition to generate a rotation matrix along X axis.

    Parameters
    ----------
    point        : ndarray
                   A point that is at the center of a plane.
    angles       : list
                   Rotation angles in degrees.

    Returns
    ----------
    plane        : ndarray
                   Points defining plane.
    """
    plane = np.array([
                      [10., 10., 0.],
                      [ 0., 10., 0.],
                      [ 0.,  0., 0.]
                     ])
    for i in range(0,plane.shape[0]):
        plane[i],_,_,_  = rotate_point(plane[i],angles=angles)
        plane[i]        = plane[i]+point
    return plane

def angle_to_radians(angle):
    """
    Definition to convert angles to radians.

    Parameters
    ----------
    angle        : float
                   Angle in degrees.

    Returns
    ----------
    radians      : float
                   Angle in radians.
    """
    radians = np.float(angle)*np.pi/180.
    return radians

def rotmatx(angle):
    """
    Definition to generate a rotation matrix along X axis.

    Parameters
    ----------
    angles       : list
                   Rotation angles in degrees.

    Returns
    ----------
    rotx         : ndarray
                    Rotation matrix along X axis.
    """
    angle = np.float(angle)
    angle = angle_to_radians(angle)
    rotx  = np.array([
                      [1.,            0.  ,           0.],
                      [0.,  math.cos(angle), -math.sin(angle)],
                      [0.,  math.sin(angle), math.cos(angle)]
                     ],dtype=np.float)
    return rotx

def rotmaty(angle):
    """
    Definition to generate a rotation matrix along Y axis.

    Parameters
    ----------
    angles       : list
                   Rotation angles in degrees.

    Returns
    ----------
    roty         : ndarray
                   Rotation matrix along Y axis.
    """
    angle = angle_to_radians(angle)
    roty  = np.array([
                      [math.cos(angle),  0., math.sin(angle)],
                      [0.,             1.,            0.],
                      [-math.sin(angle), 0., math.cos(angle)]
                     ],dtype=np.float)
    return roty

def rotmatz(angle):
    """
    Definition to generate a rotation matrix along Z axis.

    Parameters
    ----------
    angles       : list
                   Rotation angles in degrees.

    Returns
    ----------
    rotz         : ndarray
                   Rotation matrix along Z axis.
    """
    angle = angle_to_radians(angle)
    rotz  = np.array([
                      [ math.cos(angle), -math.sin(angle), 0.],
                      [ math.sin(angle),  math.cos(angle), 0.],
                      [            0.,            0., 1.]
                     ],dtype=np.float)
    return rotz

def rotate_point(point,angles=[0,0,0],mode='XYZ'):
    """
    Definition to rotate a given point.

    Parameters
    ----------
    point        : ndarray
                   A point.
    angles       : list
                   Rotation angles in degrees.
    mode         : str
                   Rotation mode determines ordering of the rotations at each axis. There are XYZ,YXZ,ZXY and ZYX modes.

    Returns
    ----------
    result       : ndarray
                   Result of the rotation
    rotx         : ndarray
                   Rotation matrix along X axis.
    roty         : ndarray
                   Rotation matrix along Y axis.
    rotz         : ndarray
                   Rotation matrix along Z axis.
    """
    rotx   = rotmatx(angles[0])
    roty   = rotmaty(angles[1])
    rotz   = rotmatz(angles[2])
    if mode == 'XYZ':
        result = np.dot(rotz,np.dot(roty,np.dot(rotx,point)))
    elif mode == 'XZY':
        result = np.dot(roty,np.dot(rotz,np.dot(rotx,point)))
    elif mode == 'YXZ':
        result = np.dot(rotz,np.dot(rotx,np.dot(roty,point)))
    elif mode == 'ZXY':
        result = np.dot(roty,np.dot(rotx,np.dot(rotz,point)))
    elif mode == 'ZYX':
        result = np.dot(rotx,np.dot(roty,np.dot(rotz,point)))
    return result,rotx,roty,rotz

if __name__ == '__main__':
    pass
