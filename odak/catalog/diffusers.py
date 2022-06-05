import numpy as np
import odak.catalog
from ..raytracing.primitives import define_plane, bring_plane_to_origin
from ..raytracing.boundary import intersect_w_surface
from ..wave import wavenumber
from ..tools.sample import circular_uniform_sample, grid_sample
from ..tools.transformation import rotate_points, tilt_towards
from ..raytracing.ray import create_ray_from_two_points


class thin_diffuser():
    """
    A class to represent a thin diffuser. This is generally useful for raytracing and wave calculations.
    """

    def __init__(self, phase=None, shape=[10., 10.], center=[0., 0., 0.], angles=[0., 0., 0.], diffusion_angle=5., diffusion_no=[3, 3], name='diffuser'):
        """
        Class to represent a simple thin diffuser with a random phase.

        Parameters
        ----------
        phase            : ndarray
                           Initial phase to be loaded. If non provided, it will start with a random phase.
        shape            : list
                           Shape of the detector.
        center           : list
                           Center of the detector.
        angles           : list
                           Rotation angles of the detector in degrees.
        diffusion angles : list
                           Full angle of diffusion along two axes.
        diffusion_no     : list
                           Number of rays to be generated along two axes at each diffusion.
        """
        self.settings = {
            'name': name,
            'center': center,
            'angles': angles,
            'rotation mode': 'XYZ',
            'shape': shape,
            'diffusion angle': diffusion_angle,
            'number of diffusion rays': diffusion_no
        }
        self.plane = define_plane(
            self.settings['center'],
            angles=self.settings['angles']
        )
        self.k = wavenumber(0.05)
        self.diffusion_points = circular_uniform_sample(
            no=self.settings["number of diffusion rays"],
            radius=np.tan(np.radians(self.settings["diffusion angle"]/2.)),
            center=[0., 0., 1.]
        )
        if type(phase) == type(None):
            self.surface_points = grid_sample(
                no=[100, 100],
                size=self.settings["shape"],
                center=self.settings["center"],
                angles=self.settings["angles"]
            )

    def plot_diffuser(self, figure):
        """
        Definition to plot diffuser to a odak.visualize.plotly.rayshow().

        Parameters
        ----------
        figure      : odak.visualize.plotly.rayshow()
                      Figure to plot the diffuser.
        """
        points = grid_sample(
            no=[10, 10],
            size=self.settings["shape"],
            center=self.settings["center"],
            angles=self.settings["angles"]
        )
        points = points.reshape((10, 10, 3))
        figure.add_surface(
            data_x=points[:, :, 0],
            data_y=points[:, :, 1],
            data_z=points[:, :, 2],
            surface_color=np.zeros(points[:, :, 2].shape),
            opacity=0.5,
            contour=False
        )

    def raytrace(self, ray):
        """
        Definition to raytrace the diffuser.


        Parameters
        ----------
        ray         : ndarray
                      Ray(s).

        Returns
        ----------
        new_rays    : ndarray
                      Diffusion rays.
        normal      : ndarray
                      Surface normals
        distance    : ndarray 
                      Distance ray(s) has/have travelled until to the point of diffusion.
        """
        if len(ray.shape) == 2:
            ray = ray.reshape((1, ray.shape[0], ray.shape[1]))
        normal, distance = intersect_w_surface(ray, self.plane)
        ########################################################################
        # This is the bottleneck for has to be replaced for better performance #
        ########################################################################
        new_rays = np.empty((0, 2, 3))
        for point_id in range(0, normal.shape[0]):
            tilt_angles = tilt_towards(ray[point_id, 0], normal[point_id, 0])
            tilted_points = rotate_points(
                self.diffusion_points,
                angles=tilt_angles,
                offset=normal[point_id, 0]
            )
            new_rays = np.vstack(
                (new_rays, create_ray_from_two_points(normal[point_id, 0], tilted_points)))
        new_rays = np.asarray(new_rays)
        return new_rays, normal, distance
