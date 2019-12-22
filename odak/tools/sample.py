from odak import np
from .transformation import rotate_point
from odak.raytracing import create_ray_from_two_points

def grid_sample(no=[10,10],size=[100.,100.],center=[0.,0.,0.],angles=[0.,0.,0.]):
    """
    Definition to generate samples over a surface.

    Parameters
    ----------
    no          : list
                  Number of samples.
    size        : list
                  Physical size of the surface.
    center      : list
                  Center location of the surface.
    angles      : list
                  Tilt of the surface.

    Returns
    ----------
    samples     : ndarray
                  Samples generated.
    """
    samples = np.zeros((no[0],no[1],3))
    step    = [
               size[0]/no[0],
               size[1]/no[1]
              ]
    for i in range(no[0]):
        for j in range(no[1]):
            point           = np.array(
                                       [
                                        i*step[0]+step[0]/2.-size[0]/2.,
                                        j*step[1]+step[1]/2.-size[1]/2.,
                                        0.
                                       ]
                                      )
            point,_,_,_     = rotate_point(
                                           point,
                                           angles=angles,
                                          )
            point[0]       += center[0]
            point[1]       += center[1]
            point[2]       += center[2]
            samples[i,j,:]  = point
    samples = samples.reshape((samples.shape[0]*samples.shape[1],samples.shape[2]))
    return samples

def batch_of_rays(entry,exit):
    """
    Definition to generate a batch of rays with given entry point(s) and exit point(s). Note that the mapping is one to one, meaning nth item in your entry points list will exit from nth item in your exit list and generate that particular ray. Note that you can have a combination like nx3 points for entry or exit and 1 point for entry or exit. But if you have multiple points both for entry and exit, the number of points have to be same both for entry and exit.

    Parameters
    ----------
    entry      : ndarray
                 Either a single point with size of 3 or multiple points with the size of nx3.
    exit       : ndarray
                 Either a single point with size of 3 or multiple points with the size of nx3.

    Returns
    ----------
    rays       : ndarray
                 Generated batch of rays.
    """
    norays = [0,0]
    if len(entry.shape) == 1:
        entry = entry.reshape((1,3))
    if len(exit.shape) == 1:
        exit = exit.reshape((1,3))
    norays = np.amax([entry.shape[0],exit.shape[0]])
    if norays > exit.shape[0]:
        exit = np.repeat(exit,norays,axis=0)
    elif norays > entry.shape[0]:
        entry = np.repeat(entry,norays,axis=0)
    rays   = []
    for i in range(norays):
        rays.append(
                    create_ray_from_two_points(
                                               entry[i],
                                               exit[i]
                                              )
                   )
    rays   = np.asarray(rays)
    return rays
