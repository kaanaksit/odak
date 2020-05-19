from odak import np
import os
import odak.catalog
import odak.tools as tools
import odak.raytracing as raytracer

class plano_convex_lens():
    def __init__(self,item='LA1024'):
        """
        Class to represent plano-convex lens.

        Parameters
        ----------
        item        : str
                      Plano convex lens label. Check Thorlabs catalog for labels. There are also json files within library.
        """
        self.path_to_catalog = "{}/plano_convex_lenses.json".format(os.path.dirname(odak.catalog.__file__))
        self.settings        = tools.load_dictionary(self.path_to_catalog)

    def define_geometry(self):
        """
        A definition to define geometry of a plano-convex lens.
        """
        self.convex_surface = tools.define_sphere(self.center,self.radius)
        return True
