import torch
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
        self.resolution = resolution.to(self.device)
        self.surface_center = center.to(self.device)
        self.surface_tilt = tilt.to(self.device)
        self.size = size.to(self.device)
        self.pixel_size = torch.tensor([
                                        self.size[0] / self.resolution[0],
                                        self.size[1] / self.resolution[1]
                                       ], device  = self.device)
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
        self.image = torch.zeros(
                                 colors,
                                 self.resolution[0],
                                 self.resolution[1],
                                 device = self.device,
                                )


    def intersect(self, rays):
        """
        Function to intersect rays with the detector


        Parameters
        ----------
        rays            : torch.tensor
                          Rays to be intersected with a detector.
                          Expected size is [1 x 2 x 3] or [m x 2 x 3].

        Returns
        -------
        image           : torch.tensor
                          Image on the image sensor [3 x k x l].
        """
        normals, _ = intersect_w_surface(rays, self.plane)
        points = normals[:, 0]
        print(self.pixel_locations.shape, points.shape)
        return self.image


