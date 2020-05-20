from odak import np
import os
import odak.catalog
import odak.tools as tools
import odak.raytracing as raytracer

class plano_convex_lens():
    def __init__(self,item='LA1024',location=[0.,0.,0.],rotation=[0.,0.,0.]):
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
        """
        self.item            = item
        self.location        = np.asarray(location)
        self.rotation        = np.asarray(rotation)
        self.path_to_catalog = "{}/data/plano_convex_lenses.json".format(os.path.dirname(odak.catalog.__file__))
        self.settings        = tools.load_dictionary(self.path_to_catalog)[self.item]
        self.set_variables()
        self.define_geometry()
    
    def set_variables(self):
        """
        A definition to set variables for definining plane-convex lens.
        """
        self.type               = self.settings["type"]
        self.units              = self.settings["units"]
        self.diameter           = self.settings["diameter"]
        self.thickness          = self.settings["center thickness"]
        self.radius             = self.settings["radius of curvature"]
        self.coating            = self.settings["coating"]
        self.center             = np.array([
                                            0.,
                                            0.,
                                            self.radius-self.thickness 
                                           ])
        self.center,_,_,_       = tools.rotate_point(
                                                     self.center,
                                                     angles=self.rotation,
                                                     offset=self.location
                                                    )
        self.plane_center       = np.array([0.,0.,0.])

    def define_geometry(self):
        """
        A definition to define geometry of a plano-convex lens.
        """
        self.convex_surface = raytracer.define_sphere(
                                                      self.center,
                                                      self.radius
                                                     )
        self.plane_surface  = raytracer.define_circle(
                                                      self.plane_center,
                                                      self.diameter,
                                                      self.rotation
                                                     )
