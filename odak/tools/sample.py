import numpy as np
from .transformation import rotate_points, rotate_point
from ..raytracing import create_ray_from_two_points


def random_sample_point_cloud(point_cloud, no, p=None):
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
    choice = np.random.choice(point_cloud.shape[0], no, p)
    subset = point_cloud[choice, :]
    return subset


def sphere_sample(no=[10, 10], radius=1., center=[0., 0., 0.], k=[1, 2]):
    """
    Definition to generate a regular sample set on the surface of a sphere using polar coordinates.

    Parameters
    ----------
    no          : list
                  Number of samples.
    radius      : float
                  Radius of a sphere.
    center      : list
                  Center of a sphere.
    k           : list
                  Multipliers for gathering samples. If you set k=[1,2] it will draw samples from a perfect sphere.

    Returns
    ----------
    samples     : ndarray
                  Samples generated.
    """
    samples = np.zeros((no[0], no[1], 3))
    psi, teta = np.mgrid[0:no[0], 0:no[1]]
    psi = k[0]*np.pi/no[0]*psi
    teta = k[1]*np.pi/no[1]*teta
    samples[:, :, 0] = center[0]+radius*np.sin(psi)*np.cos(teta)
    samples[:, :, 1] = center[0]+radius*np.sin(psi)*np.sin(teta)
    samples[:, :, 2] = center[0]+radius*np.cos(psi)
    samples = samples.reshape((no[0]*no[1], 3))
    return samples


def sphere_sample_uniform(no=[10, 10], radius=1., center=[0., 0., 0.], k=[1, 2]):
    """
    Definition to generate an uniform sample set on the surface of a sphere using polar coordinates.

    Parameters
    ----------
    no          : list
                  Number of samples.
    radius      : float
                  Radius of a sphere.
    center      : list
                  Center of a sphere.
    k           : list
                  Multipliers for gathering samples. If you set k=[1,2] it will draw samples from a perfect sphere.


    Returns
    ----------
    samples     : ndarray
                  Samples generated.

    """
    samples = np.zeros((no[0], no[1], 3))
    row = np.arange(0, no[0])
    psi, teta = np.mgrid[0:no[0], 0:no[1]]
    for psi_id in range(0, no[0]):
        psi[psi_id] = np.roll(row, psi_id, axis=0)
        teta[psi_id] = np.roll(row, -psi_id, axis=0)
    psi = k[0]*np.pi/no[0]*psi
    teta = k[1]*np.pi/no[1]*teta
    samples[:, :, 0] = center[0]+radius*np.sin(psi)*np.cos(teta)
    samples[:, :, 1] = center[1]+radius*np.sin(psi)*np.sin(teta)
    samples[:, :, 2] = center[2]+radius*np.cos(psi)
    samples = samples.reshape((no[0]*no[1], 3))
    return samples


def box_volume_sample(no=[10, 10, 10], size=[100., 100., 100.], center=[0., 0., 0.], angles=[0., 0., 0.]):
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
    samples = np.zeros((no[0], no[1], no[2], 3))
    x, y, z = np.mgrid[0:no[0], 0:no[1], 0:no[2]]
    step = [
        size[0]/no[0],
        size[1]/no[1],
        size[2]/no[2]
    ]
    samples[:, :, :, 0] = x*step[0]+step[0]/2.-size[0]/2.
    samples[:, :, :, 1] = y*step[1]+step[1]/2.-size[1]/2.
    samples[:, :, :, 2] = z*step[2]+step[2]/2.-size[2]/2.
    samples = samples.reshape(
        (samples.shape[0]*samples.shape[1]*samples.shape[2], samples.shape[3]))
    samples = rotate_points(samples, angles=angles, offset=center)
    return samples


def circular_sample(no=[10, 10], radius=10., center=[0., 0., 0.], angles=[0., 0., 0.]):
    """
    Definition to generate samples inside a circle over a surface.

    Parameters
    ----------
    no          : list
                  Number of samples.
    radius      : float
                  Radius of the circle.
    center      : list
                  Center location of the surface.
    angles      : list
                  Tilt of the surface.

    Returns
    ----------
    samples     : ndarray
                  Samples generated.
    """
    samples = np.zeros((no[0]+1, no[1]+1, 3))
    r_angles, r = np.mgrid[0:no[0]+1, 0:no[1]+1]
    r = r/np.amax(r)*radius
    r_angles = r_angles/np.amax(r_angles)*np.pi*2
    samples[:, :, 0] = r*np.cos(r_angles)
    samples[:, :, 1] = r*np.sin(r_angles)
    samples = samples[1:no[0]+1, 1:no[1]+1, :]
    samples = samples.reshape(
        (samples.shape[0]*samples.shape[1], samples.shape[2]))
    samples = rotate_points(samples, angles=angles, offset=center)
    return samples


def circular_uniform_random_sample(no=[10, 50], radius=10., center=[0., 0., 0.], angles=[0., 0., 0.]):
    """ 
    Definition to generate sample inside a circle uniformly but randomly.

    Parameters
    ----------
    no          : list
                  Number of samples.
    radius      : float
                  Radius of the circle.
    center      : list
                  Center location of the surface.
    angles      : list
                  Tilt of the surface.

    Returns
    ----------
    samples     : ndarray
                  Samples generated.
    """
    samples = np.empty((0, 3))
    rs = np.sqrt(np.random.uniform(0, 1, no[0]))
    angs = np.random.uniform(0, 2*np.pi, no[1])
    for i in rs:
        for angle in angs:
            r = radius*i
            point = np.array(
                [float(r*np.cos(angle)), float(r*np.sin(angle)), 0])
            samples = np.vstack((samples, point))
    samples = rotate_points(samples, angles=angles, offset=center)
    return samples


def circular_uniform_sample(no=[10, 50], radius=10., center=[0., 0., 0.], angles=[0., 0., 0.]):
    """
    Definition to generate sample inside a circle uniformly.

    Parameters
    ----------
    no          : list
                  Number of samples.
    radius      : float
                  Radius of the circle.
    center      : list
                  Center location of the surface.
    angles      : list
                  Tilt of the surface.

    Returns
    ----------
    samples     : ndarray
                  Samples generated.
    """
    samples = np.empty((0, 3))
    for i in range(0, no[0]):
        r = i/no[0]*radius
        ang_no = no[1]*i/no[0]
        for j in range(0, int(no[1]*i/no[0])):
            angle = j/ang_no*2*np.pi
            point = np.array(
                [float(r*np.cos(angle)), float(r*np.sin(angle)), 0])
            samples = np.vstack((samples, point))
    samples = rotate_points(samples, angles=angles, offset=center)
    return samples


def grid_sample(no=[10, 10], size=[100., 100.], center=[0., 0., 0.], angles=[0., 0., 0.]):
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
    samples = np.zeros((no[0], no[1], 3))
    step = [
        size[0]/(no[0]-1),
        size[1]/(no[1]-1)
    ]
    x, y = np.mgrid[0:no[0], 0:no[1]]
    samples[:, :, 0] = x*step[0]-size[0]/2.
    samples[:, :, 1] = y*step[1]-size[1]/2.
    samples = samples.reshape(
        (samples.shape[0]*samples.shape[1], samples.shape[2]))
    samples = rotate_points(samples, angles=angles, offset=center)
    return samples


def batch_of_rays(entry, exit):
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
    norays = np.array([0, 0])
    if len(entry.shape) == 1:
        entry = entry.reshape((1, 3))
    if len(exit.shape) == 1:
        exit = exit.reshape((1, 3))
    norays = np.amax(np.asarray([entry.shape[0], exit.shape[0]]))
    if norays > exit.shape[0]:
        exit = np.repeat(exit, norays, axis=0)
    elif norays > entry.shape[0]:
        entry = np.repeat(entry, norays, axis=0)
    rays = []
    norays = int(norays)
    for i in range(norays):
        rays.append(
            create_ray_from_two_points(
                entry[i],
                exit[i]
            )
        )
    rays = np.asarray(rays)
    return rays
