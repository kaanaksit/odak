import os
import numpy as np
import odak
from ..tools.transformation import rotate_point
from ..tools.vector import same_side
from ..tools.file import load_dictionary
from ..raytracing.primitives import define_sphere, define_circle
from ..raytracing.boundary import intersect_w_circle, intersect_w_sphere


class plano_convex_lens():
    """
    A class to represent a plano-convex lens. This is generally useful for raytracing and wave calculations.
    """

    def __init__(self, item='LA1024', location=[0., 0., 0.], rotation=[0., 0., 0.], wavelength=0.000532, meduium='air'):
        """
        Class to represent plano-convex lens.

        Parameters
        ----------
        item        : str
                      Plano convex lens label. Check Thorlabs catalog for labels. There are also json files within library.
        location    : list
                      Location in X,Y, and Z.
        rotation    : list
                      Rotation in X,Y, and Z.
        wavelength  : float
                      Wavelength in mm (default).
        medium      : str
                      Medium that the lens is in. Default is air.
        """
        self.item = item
        self.location = np.asarray(location)
        self.rotation = np.asarray(rotation)
        self.path_to_catalog = "{}/data/plano_convex_lenses.json".format(
            os.path.dirname(odak.catalog.__file__))
        self.settings = load_dictionary(self.path_to_catalog)[self.item]
        self.set_variables()
        self.define_geometry()

    def set_variables(self):
        """
        A definition to set variables for definining plane-convex lens.
        """
        self.type = self.settings["type"]
        self.units = self.settings["units"]
        self.diameter = self.settings["diameter"]
        self.thickness = self.settings["center thickness"]
        self.radius = self.settings["radius of curvature"]
        self.coating = self.settings["coating"]

    def define_geometry(self):
        """
        A definition to define geometry of a plano-convex lens.
        """
        self.center = np.array([
            0.,
            0.,
            self.radius-self.thickness
        ])
        self.center, _, _, _ = rotate_point(
            self.center,
            angles=self.rotation,
            offset=self.location
        )

        self.plane_center = np.array([0., 0., 0.])+self.location
        self.convex_point = self.plane_center-self.thickness
        self.convex_surface = define_sphere(
            self.center,
            self.radius
        )
        self.plane_surface = define_circle(
            self.plane_center,
            self.diameter,
            self.rotation
        )

    def intersect(self, ray):
        """
        A  definition to find out which surface to intersect a ray(s).

        Parameters
        ----------
        ray          : ndarray
                       Ray(s) to be intersected. 
        """
        convex_normal, convex_distance = intersect_w_sphere(
            ray,
            self.convex_surface
        )
        plane_normal, plane_distance = intersect_w_circle(
            ray,
            self.plane_surface
        )
        test_normal = convex_normal
        if len(test_normal.shape) < 3:
            test_normal = convex_normal[0]
        is_it_in_lens = same_side(
            test_normal,
            self.convex_point,
            self.plane_surface[0][1],
            self.plane_surface[0][2]
        )
        surface_normals = np.array(
            [convex_normal, plane_normal], dtype=np.float64)
        surface_distances = np.array(
            [convex_distance, plane_distance], dtype=np.float64)
        which_surface = np.amin(surface_distances, axis=0)
        ids = np.where(surface_distances == which_surface)
        ids = np.asarray(ids)
#        ids[0]                        = ids[0] | is_it_in_lens
        normal = surface_normals[ids[0], ids[1]]
        distance = surface_distances[ids[0], ids[1]]
        return normal, distance

    def raytrace(self):
        """
        A definition to raytrace input ray(s) with the plano-convex lens.


        Parameters
        ----------
        rays       : ndarray
                     Input ray(s).
        """
#        which_surface
#        refract
#        return output_rays
        return None
