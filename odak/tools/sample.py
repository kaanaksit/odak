from odak import np
from .transformation import rotate_point
from odak.raytracing import create_ray_from_two_points


def random_sample_point_cloud(point_cloud,no,p=None):
    """
    Definition to pull a subset of points from a point cloud with a given probability.

    Parameters
    ----------
    point_cloud  : ndarray
                   Point cloud array.
    no           : list
                   Number of samples.
    p            : list
                   Probability list in the same size as no.

    Returns
    ----------
    subset       : ndarray
                   Subset of the given point cloud.
    """
    choice = np.random.choice(point_cloud.shape[0],no,p)
    subset = point_cloud[choice,:]
    return subset 


def sphere_sample(no=[10,10],radius=1.,center=[0.,0.,0.]):
    """
    Definition to generate a regular sample set on the surface of a sphere.

    Parameters
    ----------
    no          : list
                  Number of samples.
    radius      : float
                  Radius of a sphere.
    center      : list
                  Center of a sphere.

    Returns
    ----------
    samples     : ndarray
                  Samples generated.
    """
    samples        = np.zeros((no[0],no[1],3))
    psi,teta       = np.mgrid[0:no[0],0:no[1]]
    psi            = 2*np.pi/no[0]*psi
    teta           = 2*np.pi/no[1]*teta
    samples[:,:,0] = center[0]+radius*np.sin(psi)*np.cos(teta)
    samples[:,:,1] = center[0]+radius*np.sin(psi)*np.sin(teta)
    samples[:,:,2] = center[0]+radius*np.cos(psi)
    samples        = samples.reshape((no[0]*no[1],3))
    return samples

def box_volume_sample(no=[10,10,10],size=[100.,100.,100.],center=[0.,0.,0.],angles=[0.,0.,0.]):
    """
    Definition to generate samples in a box volume.

    Parameters
    ----------
    no          : list
                  Number of samples.
    size        : list
                  Physical size of the volume.
    center      : list
                  Center location of the volume.
    angles      : list
                  Tilt of the volume.

    Returns
    ----------
    samples     : ndarray
                  Samples generated.
    """
    samples = np.zeros((no[0],no[1],no[2],3))
    step    = [
               size[0]/no[0],
               size[1]/no[1],
               size[2]/no[2]
              ]
    for i in range(no[0]):
        for j in range(no[1]):
            for k in range(no[2]):
                point             = np.array(
                                             [
                                              i*step[0]+step[0]/2.-size[0]/2.,
                                              j*step[1]+step[1]/2.-size[1]/2.,
                                              k*step[2]+step[2]/2.-size[2]/2.
                                             ]
                                            )
                point,_,_,_       = rotate_point(
                                                 point,
                                                 angles=angles,
                                                )
                point[0]         += center[0]
                point[1]         += center[1]
                point[2]         += center[2]
                samples[i,j,k,:]  = point
    samples = samples.reshape((samples.shape[0]*samples.shape[1]*samples.shape[2],samples.shape[3]))
    return samples

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
    norays = np.array([0,0])
    if len(entry.shape) == 1:
        entry = entry.reshape((1,3))
    if len(exit.shape) == 1:
        exit = exit.reshape((1,3))
    norays = np.amax(np.asarray([entry.shape[0],exit.shape[0]]))
    if norays > exit.shape[0]:
        exit = np.repeat(exit,norays,axis=0)
    elif norays > entry.shape[0]:
        entry = np.repeat(entry,norays,axis=0)
    rays   = []
    norays = int(norays)
    for i in range(norays):
        rays.append(
                    create_ray_from_two_points(
                                               entry[i],
                                               exit[i]
                                              )
                   )
    rays   = np.asarray(rays)
    return rays
