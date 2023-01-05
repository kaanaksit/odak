from numpy import dot
import torch
import math


def make_3d_location_map(image_pixel_size, real_image_width=0.3, real_viewing_distance=0.6):
    """ 
    Makes a map of the real 3D location that each pixel in an image corresponds to, when displayed to
    a user on a flat screen. Assumes the viewpoint is located at the centre of the image, and the screen is 
    perpendicular to the viewing direction.

    Parameters
    ----------

    image_pixel_size        : tuple of ints 
                                The size of the image in pixels, as a tuple of form (height, width)
    real_image_width        : float
                                The real width of the image as displayed. Units not important, as long as they
                                are the same as those used for real_viewing_distance
    real_viewing_distance   : float 
                                The real distance from the user's viewpoint to the screen.

    Returns
    -------

    map                     : torch.tensor
                                The computed 3D location map, of size 3xWxH.
    """
    real_image_height = (real_image_width /
                         image_pixel_size[-1]) * image_pixel_size[-2]
    x_coords = torch.linspace(-0.5, 0.5, image_pixel_size[-1])*real_image_width
    x_coords = x_coords[None, None, :].repeat(1, image_pixel_size[-2], 1)
    y_coords = torch.linspace(-0.5, 0.5,
                              image_pixel_size[-2])*real_image_height
    y_coords = y_coords[None, :, None].repeat(1, 1, image_pixel_size[-1])
    z_coords = torch.ones(
        (1, image_pixel_size[-2], image_pixel_size[-1])) * real_viewing_distance

    return torch.cat([x_coords, y_coords, z_coords], dim=0)


def make_eccentricity_distance_maps(gaze_location, image_pixel_size, real_image_width=0.3, real_viewing_distance=0.6):
    """ 
    Makes a map of the eccentricity of each pixel in an image for a given fixation point, when displayed to
    a user on a flat screen. Assumes the viewpoint is located at the centre of the image, and the screen is 
    perpendicular to the viewing direction. Output in radians.

    Parameters
    ----------

    gaze_location           : tuple of floats
                                User's gaze (fixation point) in the image. Should be given as a tuple with normalized
                                image coordinates (ranging from 0 to 1)
    image_pixel_size        : tuple of ints
                                The size of the image in pixels, as a tuple of form (height, width)
    real_image_width        : float
                                The real width of the image as displayed. Units not important, as long as they
                                are the same as those used for real_viewing_distance
    real_viewing_distance   : float
                                The real distance from the user's viewpoint to the screen.

    Returns
    -------

    eccentricity_map        : torch.tensor
                                The computed eccentricity map, of size WxH.
    distance_map            : torch.tensor
                                The computed distance map, of size WxH.
    """
    real_image_height = (real_image_width /
                         image_pixel_size[-1]) * image_pixel_size[-2]
    location_map = make_3d_location_map(
        image_pixel_size, real_image_width, real_viewing_distance)
    distance_map = torch.sqrt(torch.sum(location_map*location_map, dim=0))
    direction_map = location_map / distance_map

    gaze_location_3d = torch.tensor([
        (gaze_location[0]*2 - 1)*real_image_width*0.5,
        (gaze_location[1]*2 - 1)*real_image_height*0.5,
        real_viewing_distance])
    gaze_dir = gaze_location_3d / \
        torch.sqrt(torch.sum(gaze_location_3d * gaze_location_3d))
    gaze_dir = gaze_dir[:, None, None]

    dot_prod_map = torch.sum(gaze_dir * direction_map, dim=0)
    dot_prod_map = torch.clamp(dot_prod_map, min=-1.0, max=1.0)
    eccentricity_map = torch.acos(dot_prod_map)

    return eccentricity_map, distance_map


def make_pooling_size_map_pixels(gaze_location, image_pixel_size, alpha=0.3, real_image_width=0.3, real_viewing_distance=0.6, mode="quadratic"):
    """ 
    Makes a map of the pooling size associated with each pixel in an image for a given fixation point, when displayed to
    a user on a flat screen. Follows the idea that pooling size (in radians) should be directly proportional to eccentricity
    (also in radians). 

    Assumes the viewpoint is located at the centre of the image, and the screen is 
    perpendicular to the viewing direction. Output is the width of the pooling region in pixels.

    Parameters
    ----------

    gaze_location           : tuple of floats
                                User's gaze (fixation point) in the image. Should be given as a tuple with normalized
                                image coordinates (ranging from 0 to 1)
    image_pixel_size        : tuple of ints
                                The size of the image in pixels, as a tuple of form (height, width)
    real_image_width        : float
                                The real width of the image as displayed. Units not important, as long as they
                                are the same as those used for real_viewing_distance
    real_viewing_distance   : float
                                The real distance from the user's viewpoint to the screen.

    Returns
    -------

    pooling_size_map        : torch.tensor
                                The computed pooling size map, of size WxH.
    """
    eccentricity, distance_to_pixel = make_eccentricity_distance_maps(
        gaze_location, image_pixel_size, real_image_width, real_viewing_distance)
    eccentricity_centre, _ = make_eccentricity_distance_maps(
        [0.5, 0.5], image_pixel_size, real_image_width, real_viewing_distance)
    pooling_rad = alpha * eccentricity
    if mode == "quadratic":
        pooling_rad *= eccentricity
    angle_min = eccentricity_centre - pooling_rad*0.5
    angle_max = eccentricity_centre + pooling_rad*0.5
    major_axis = (torch.tan(angle_max) - torch.tan(angle_min)) * \
        real_viewing_distance
    minor_axis = 2 * distance_to_pixel * torch.tan(pooling_rad*0.5)
    area = math.pi * major_axis * minor_axis * 0.25
    # Should be +ve anyway, but check to ensure we don't take sqrt of negative number
    area = torch.abs(area)
    pooling_real = torch.sqrt(area)
    pooling_pixel = (pooling_real / real_image_width) * image_pixel_size[1]
    return pooling_pixel


def make_pooling_size_map_lod(gaze_location, image_pixel_size, alpha=0.3, real_image_width=0.3, real_viewing_distance=0.6, mode="quadratic"):
    """ 
    This function is similar to make_pooling_size_map_pixels, but instead returns a map of LOD levels to sample from
    to achieve the correct pooling region areas.

    Parameters
    ----------

    gaze_location           : tuple of floats
                                User's gaze (fixation point) in the image. Should be given as a tuple with normalized
                                image coordinates (ranging from 0 to 1)
    image_pixel_size        : tuple of ints
                                The size of the image in pixels, as a tuple of form (height, width)
    real_image_width        : float
                                The real width of the image as displayed. Units not important, as long as they
                                are the same as those used for real_viewing_distance
    real_viewing_distance   : float
                                The real distance from the user's viewpoint to the screen.

    Returns
    -------

    pooling_size_map        : torch.tensor
                                The computed pooling size map, of size WxH.
    """
    pooling_pixel = make_pooling_size_map_pixels(
        gaze_location, image_pixel_size, alpha, real_image_width, real_viewing_distance, mode)
    pooling_lod = torch.log2(1e-6+pooling_pixel)
    pooling_lod[pooling_lod < 0] = 0
    return pooling_lod


def make_radial_map(size, gaze):
    """ 
    Makes a simple radial map where each pixel contains distance in pixels from the chosen gaze location.

    Parameters
    ----------

    size    : tuple of ints
                Dimensions of the image
    gaze    : tuple of floats
                User's gaze (fixation point) in the image. Should be given as a tuple with normalized
                image coordinates (ranging from 0 to 1)
    """
    pix_gaze = [gaze[0]*size[0], gaze[1]*size[1]]
    rows = torch.linspace(0, size[0], size[0])
    rows = rows[:, None].repeat(1, size[1])
    cols = torch.linspace(0, size[1], size[1])
    cols = cols[None, :].repeat(size[0], 1)
    dist_sq = torch.pow(rows - pix_gaze[0], 2) + \
        torch.pow(cols - pix_gaze[1], 2)
    radii = torch.sqrt(dist_sq)
    return radii/torch.max(radii)

def make_equi_pooling_size_map_pixels(gaze_angles, image_pixel_size, alpha=0.3, mode="quadratic"):
    """
    This function makes a map of pooling sizes in pixels, similarly to make_pooling_size_map_pixels, but works on 360 equirectangular images.
    Input images are assumed to be in equirectangular form - i.e. if you consider a 3D viewing setup where y is the vertical axis, 
    the x location in the image corresponds to rotation around the y axis (yaw), ranging from -pi to pi. The y location in the image
    corresponds to pitch, ranging from -pi/2 to pi/2.

    In this setup real_image_width and real_viewing_distance have no effect.

    Note that rather than a 2D image gaze location in [0,1]^2, the gaze should be specified as gaze angles in [-pi,pi]x[-pi/2,pi/2] (yaw, then pitch).

    Parameters
    ----------

    gaze_angles         : tuple of 2 floats
                            Gaze direction expressed as angles, in radians.
    image_pixel_size    : tuple of 2 ints
                            Dimensions of the image in pixels, as a tuple of (height, width)
    alpha               : float
                            Parameter controlling extent of foveation
    mode                : str
                            Foveation mode (how pooling size varies with eccentricity). Should be "quadratic" or "linear"
    """
    view_direction = torch.tensor([math.sin(gaze_angles[0])*math.cos(gaze_angles[1]), math.sin(gaze_angles[1]), math.cos(gaze_angles[0])*math.cos(gaze_angles[1])])

    yaw_angle_map = torch.linspace(-torch.pi, torch.pi, image_pixel_size[1])
    yaw_angle_map = yaw_angle_map[None,:].repeat(image_pixel_size[0], 1)[None,...]
    pitch_angle_map = torch.linspace(-torch.pi*0.5, torch.pi*0.5, image_pixel_size[0])
    pitch_angle_map = pitch_angle_map[:,None].repeat(1, image_pixel_size[1])[None,...]

    dir_map = torch.cat([torch.sin(yaw_angle_map)*torch.cos(pitch_angle_map), torch.sin(pitch_angle_map), torch.cos(yaw_angle_map)*torch.cos(pitch_angle_map)])
    
    # Work out the pooling region diameter in radians
    view_dot_dir = torch.sum(view_direction[:,None,None] * dir_map, dim=0)
    eccentricity = torch.acos(view_dot_dir)
    pooling_rad = alpha * eccentricity
    if mode == "quadratic":
        pooling_rad *= eccentricity

    # The actual pooling region will be an ellipse in the equirectangular image - the length of the major & minor axes
    # depend on the x & y resolution of the image. We find these two axis lengths (in pixels) and then the area of the ellipse
    pixels_per_rad_x = image_pixel_size[1] / (2*torch.pi)
    pixels_per_rad_y = image_pixel_size[0] / (torch.pi)
    pooling_axis_x = pooling_rad * pixels_per_rad_x
    pooling_axis_y = pooling_rad * pixels_per_rad_y
    area = torch.pi * pooling_axis_x * pooling_axis_y * 0.25

    # Now finally find the length of the side of a square of the same area.
    size = torch.sqrt(torch.abs(area))
    return size


def make_equi_pooling_size_map_lod(gaze_angles, image_pixel_size, alpha=0.3, mode="quadratic"):
    """ 
    This function is similar to make_equi_pooling_size_map_pixels, but instead returns a map of LOD levels to sample from
    to achieve the correct pooling region areas.

    Parameters
    ----------

    gaze_angles         : tuple of 2 floats
                            Gaze direction expressed as angles, in radians.
    image_pixel_size    : tuple of 2 ints
                            Dimensions of the image in pixels, as a tuple of (height, width)
    alpha               : float
                            Parameter controlling extent of foveation
    mode                : str
                            Foveation mode (how pooling size varies with eccentricity). Should be "quadratic" or "linear"

    Returns
    -------

    pooling_size_map        : torch.tensor
                                The computed pooling size map, of size HxW.
    """
    pooling_pixel = make_equi_pooling_size_map_pixels(gaze_angles, image_pixel_size, alpha, mode)
    import matplotlib.pyplot as plt
    pooling_lod = torch.log2(1e-6+pooling_pixel)
    pooling_lod[pooling_lod < 0] = 0
    return pooling_lod
