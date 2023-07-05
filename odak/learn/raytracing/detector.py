import torch
import logging
from .primitives import define_plane
from .boundary import intersect_w_surface
from ..tools import grid_sample


class detector():
    """
    A class to represent a detector.
    """


    def __init__(
                 self,
                 colors = 3,
                 center = torch.tensor([0., 0., 0.]),
                 tilt = torch.tensor([0., 0., 0.]),
                 size = torch.tensor([10., 10.]),
                 resolution = torch.tensor([100, 100]),
                 device = torch.device('cpu')
                ):
        """
        Parameters
        ----------
        colors         : int
                         Number of color channels to register (e.g., RGB).
        center         : torch.tensor
                         Center point of the detector [3].
        tilt           : torch.tensor
                         Tilt angles of the surface in degrees [3].
        size           : torch.tensor
                         Size of the detector [2].
        resolution     : torch.tensor
                         Resolution of the detector.
        device         : torch.device
                         Device for computation (e.g., cuda, cpu).
        """
        self.device = device
        self.colors = colors
        self.resolution = resolution.to(self.device)
        self.surface_center = center.to(self.device)
        self.surface_tilt = tilt.to(self.device)
        self.size = size.to(self.device)
        self.pixel_size = torch.tensor([
                                        self.size[0] / self.resolution[0],
                                        self.size[1] / self.resolution[1]
                                       ], device  = self.device)
        self.pixel_diagonal_size = torch.sqrt(self.pixel_size[0] ** 2 + self.pixel_size[1] ** 2)
        self.pixel_diagonal_half_size = self.pixel_diagonal_size / 2.
        self.threshold = torch.nn.Threshold(self.pixel_diagonal_size, 1)
        self.plane = define_plane(
                                  point = self.surface_center,
                                  angles = self.surface_tilt
                                 )
        self.pixel_locations, _, _, _ = grid_sample(
                                                    size = self.size.tolist(),
                                                    no = self.resolution.tolist(),
                                                    center = self.surface_center.tolist(),
                                                    angles = self.surface_tilt.tolist()
                                                   )
        self.pixel_locations = self.pixel_locations.to(self.device)
        self.relu = torch.nn.ReLU()
        self.clear()


    def intersect(self, rays, color = 0):
        """
        Function to intersect rays with the detector


        Parameters
        ----------
        rays            : torch.tensor
                          Rays to be intersected with a detector.
                          Expected size is [1 x 2 x 3] or [m x 2 x 3].
        color           : int
                          Color channel to register.

        Returns
        -------
        points          : torch.tensor
                          Intersection points with the image detector [k x 3].
        """
        normals, _ = intersect_w_surface(rays, self.plane)
        points = normals[:, 0]
        distances_xyz = torch.abs(points.unsqueeze(1) - self.pixel_locations.unsqueeze(0))
        distances_x = 1e6 * self.relu(- (distances_xyz[:, :, 0] - self.pixel_size[0]))
        distances_y = 1e6 * self.relu(- (distances_xyz[:, :, 1] - self.pixel_size[1]))
        hit_x = torch.clamp(distances_x, min = 0., max = 1.)
        hit_y = torch.clamp(distances_y, min = 0., max = 1.)
#        hit_x = (1. / (1. + torch.exp(1e6 * distances_x)) - 0.5) * 2.
#        hit_y = (1. / (1. + torch.exp(1e6 * distances_y)) - 0.5) * 2.
        hit = hit_x * hit_y
        image = torch.sum(hit_x * hit_y, dim = 0)
        self.image[color] += image.reshape(
                                           self.image.shape[-2], 
                                           self.image.shape[-1]
                                          )
        return points


    def get_image(self):
        """
        Function to return the detector image.

        Returns
        -------
        image           : torch.tensor
                          Detector image.
        """
        image = torch.zeros_like(self.image)
        image = (self.image - self.image.min()) / (self.image.max() - self.image.min())
        return image


    def convert_image_to_points(
                                self, 
                                image: torch.Tensor,
                                color = 0, 
                                ray_count = 100
                               ):
        """
        Extracting the locations of non-zero pixels.

        Parameters
        ----------
        image           : torch.tensor
                          Image on a detector [k x m x n].
                          Pixel values should be normalized between one and zero.
        color           : int
                          Color channel for identifying non-zero pixels.
        ray_count       : int
                          Number of rays to represent the image.

        Returns
        -------
        image_points    : torch.tensor
                          Non-zero pixel locations [j x 3].
        """
        image = image[color].reshape(-1)
        image_points = self.pixel_locations[image > 0., :]
        image_values = image[image > 0.]
        pixel_ray_count = ray_count / torch.sum(image_values)
        image_values *= pixel_ray_count
        image_values = image_values.int()
        if image_values.max() < 1.:
            logging.warning('[odak.learn.raytracing.convert_image_to_points] Number of rays are too small: {}.'.format(ray_count))
        image_points = torch.repeat_interleave(image_points, repeats = image_values, dim = 0)
        return image_points, image_values


    def clear(self):
        """
        Internal function to clear a detector.
        """
        self.image = torch.zeros(
                                 self.colors,
                                 self.resolution[0],
                                 self.resolution[1],
                                 device = self.device,
                                )

