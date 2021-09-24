from numpy import dot
import torch
import math

def make_3d_location_map(image_pixel_size, real_image_width=0.3, real_viewing_distance=0.6):
    """ Makes a map of the real 3D location that each pixel in an image corresponds to, when displayed to
        a user on a flat screen. Assumes the viewpoint is located at the centre of the image, and the screen is 
        perpendicular to the viewing direction.
        image_pixel_size: The size of the image in pixels, as a tuple of form (height, width)
        real_image_width: The real width of the image as displayed. Units not important, as long as they
            are the same as those used for real_viewing_distance
        real_viewing_distance: The real distance from the user's viewpoint to the screen.
    """
    real_image_height = (real_image_width / image_pixel_size[-1]) * image_pixel_size[-2]
    x_coords = torch.linspace(-0.5, 0.5, image_pixel_size[-1])*real_image_width
    x_coords = x_coords[None,None,:].repeat(1,image_pixel_size[-2],1)
    y_coords = torch.linspace(-0.5, 0.5, image_pixel_size[-2])*real_image_height
    y_coords = y_coords[None,:,None].repeat(1,1,image_pixel_size[-1])
    z_coords = torch.ones((1, image_pixel_size[-2], image_pixel_size[-1])) * real_viewing_distance

    return torch.cat([x_coords, y_coords, z_coords], dim=0)

def make_eccentricity_distance_maps(gaze_location, image_pixel_size, real_image_width=0.3, real_viewing_distance=0.6):
    """ Makes a map of the eccentricity of each pixel in an image for a given fixation point, when displayed to
        a user on a flat screen. Assumes the viewpoint is located at the centre of the image, and the screen is 
        perpendicular to the viewing direction. Output in radians.
        gaze_location: User's gaze (fixation point) in the image. Should be given as a tuple with normalized
            image coordinates (ranging from 0 to 1)
        image_pixel_size: The size of the image in pixels, as a tuple of form (height, width)
        real_image_width: The real width of the image as displayed. Units not important, as long as they
            are the same as those used for real_viewing_distance
        real_viewing_distance: The real distance from the user's viewpoint to the screen.
    """
    real_image_height = (real_image_width / image_pixel_size[-1]) * image_pixel_size[-2]
    location_map = make_3d_location_map(image_pixel_size, real_image_width, real_viewing_distance)
    distance_map = torch.sqrt(torch.sum(location_map*location_map, dim=0))
    direction_map = location_map / distance_map

    gaze_location_3d = torch.tensor([\
        (gaze_location[0]*2 - 1)*real_image_width*0.5,\
        (gaze_location[1]*2 - 1)*real_image_height*0.5,\
        real_viewing_distance])
    gaze_dir = gaze_location_3d / torch.sqrt(torch.sum(gaze_location_3d * gaze_location_3d))
    gaze_dir = gaze_dir[:,None,None]

    dot_prod_map = torch.sum(gaze_dir * direction_map, dim=0)
    dot_prod_map = torch.clamp(dot_prod_map, min=-1.0, max=1.0)
    eccentricity_map = torch.acos(dot_prod_map)

    return eccentricity_map, distance_map

def make_pooling_size_map_pixels(gaze_location, image_pixel_size, alpha=0.3, real_image_width=0.3, real_viewing_distance=0.6, mode="quadratic"):
    """ Makes a map of the pooling size associated with each pixel in an image for a given fixation point, when displayed to
        a user on a flat screen. Follows the idea that pooling size (in radians) should be directly proportional to eccentricity
        (also in radians). 
        Assumes the viewpoint is located at the centre of the image, and the screen is 
        perpendicular to the viewing direction. Output is the width of the pooling region in pixels.
        gaze_location: User's gaze (fixation point) in the image. Should be given as a tuple with normalized
            image coordinates (ranging from 0 to 1)
        image_pixel_size: The size of the image in pixels, as a tuple of form (height, width)
        alpha: The constant of proportionality (i.e. pooling size = alpha x eccentricity).
        real_image_width: The real width of the image as displayed. Units not important, as long as they
            are the same as those used for real_viewing_distance
        real_viewing_distance: The real distance from the user's viewpoint to the screen.
    """
    eccentricity, distance_to_pixel = make_eccentricity_distance_maps(gaze_location, image_pixel_size, real_image_width, real_viewing_distance)
    eccentricity_centre, _ = make_eccentricity_distance_maps([0.5,0.5], image_pixel_size, real_image_width, real_viewing_distance)
    pooling_rad = alpha * eccentricity
    if mode == "quadratic":
        pooling_rad *= eccentricity
    angle_min = eccentricity_centre - pooling_rad*0.5
    angle_max = eccentricity_centre + pooling_rad*0.5
    major_axis = (torch.tan(angle_max) - torch.tan(angle_min)) / real_viewing_distance
    minor_axis = 2 * distance_to_pixel * torch.tan(pooling_rad*0.5)
    area = math.pi * major_axis * minor_axis
    area = torch.abs(area) #Should be +ve anyway, but check to ensure we don't take sqrt of negative number
    pooling_real = torch.sqrt(area)
    pooling_pixel = (pooling_real / real_image_width) * image_pixel_size[1]
    return pooling_pixel

def make_pooling_size_map_lod(gaze_location, image_pixel_size, alpha=0.3, real_image_width=0.3, real_viewing_distance=0.6, mode="quadratic"):
    """ Like make_pooling_size_map_pixels, but instead gives the pooling sizes as LOD levels of a mipmap to sample from.
    """
    pooling_pixel = make_pooling_size_map_pixels(gaze_location, image_pixel_size, alpha, real_image_width, real_viewing_distance, mode)
    pooling_lod = torch.log2(1e-6+pooling_pixel)
    pooling_lod[pooling_lod < 0] = 0
    return pooling_lod

def make_radial_map(size, gaze):
    """ Makes a simple radial map where each pixel contains distance in pixels from the chosen gaze location.
        Gaze location should be supplied in normalised image coordinates.
    """
    pix_gaze = [gaze[0]*size[0], gaze[1]*size[1]]
    rows = torch.linspace(0, size[0], size[0])
    rows = rows[:,None].repeat(1, size[1])
    cols = torch.linspace(0, size[1], size[1])
    cols = cols[None,:].repeat(size[0], 1)
    dist_sq = torch.pow(rows - pix_gaze[0], 2) + torch.pow(cols - pix_gaze[1], 2)
    radii = torch.sqrt(dist_sq)
    return radii/torch.max(radii)