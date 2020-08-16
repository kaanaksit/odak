from odak import np
import os
import odak.catalog
from odak.raytracing.primitives import define_plane,bring_plane_to_origin
from odak.raytracing.boundary import intersect_w_surface

class plane_detector():
    def __init__(self,field=None,resolution=[1000,1000],shape=[10.,10.],center=[0.,0.,0.],angles=[0.,0.,0.]):
        """
        Class to represent a simple planar detector.

        Parameters
        ----------
        """
        self.settings   = {
                           'resolution'    : resolution,
                           'center'        : center,
                           'angles'        : angles,
                           'rotation mode' : 'XYZ',
                           'shape'         : shape,
                          }
        self.plane      = define_plane(
                                       self.settings['center'],
                                       angles=self.settings['angles']
                                      )
        if type(field) == type(None):
            self.field = np.zeros(
                                  (
                                   self.settings['resolution'][0],
                                   self.settings['resolution'][1],
                                   1
                                  ),
                                  dtype=np.complex64
                                  )

    def raytrace(self,ray,field=1,channel=0):
        """
        A definition to calculate the intersection between given ray(s) and the detector. If a ray contributes to the detector, field will be taken into account in calculating the field over the planar detector.
 
        Parameters
        ----------
        ray          : ndarray
                       Ray(s) to be intersected.
        field        : ndarray
                       Field(s) to be used for calculating contribution of rays to the detector.
        channel      : list
                       Which color channel to contribute to in the detector plane. Default is zero.
 
        Returns
        ----------
        normal       : ndarray
                       Normal for each intersection point.
        distance     : ndarray
                       Distance for each ray.
        """
        normal,distance = intersect_w_surface(ray,self.plane)
        points          = bring_plane_to_origin(
                                                normal[:,0],
                                                self.plane,
                                                shape=self.settings["shape"],
                                                center=self.settings["center"],
                                                angles=self.settings["angles"],
                                                mode=self.settings["rotation mode"]
                                               )
        if points.shape[0] == 3:
            points = points.reshape((1,3))
        # This could improve with a bilinear filter. Basically removing int with a filter.
        detector_ids = np.array(
                                [
                                 (points[:,0]+self.settings["shape"][0]/2.)/self.settings["shape"][0]*self.settings["resolution"][0]+1,
                                 (points[:,1]+self.settings["shape"][1]/2.)/self.settings["shape"][1]*self.settings["resolution"][1]+1
                                ],
                                dtype=int
                               )
        detector_ids[0,:] = (detector_ids[0,:]>=1)*detector_ids[0,:]
        detector_ids[1,:] = (detector_ids[1,:]>=1)*detector_ids[1,:]
        detector_ids[0,:] = (detector_ids[0,:]<=self.settings["resolution"][0]+1)*detector_ids[0,:]
        detector_ids[1,:] = (detector_ids[1,:]<=self.settings["resolution"][1]+1)*detector_ids[1,:]
        cache             = np.zeros(
                                     (
                                      self.settings["resolution"][0]+1,
                                      self.settings["resolution"][1]+1,
                                      self.field.shape[2]
                                     ),
                                     dtype=np.complex64
                                    )
        cache[
              detector_ids[0],
              detector_ids[1],
              channel
             ]           += field
        self.field       += cache[1::,1::,:]
        return normal,distance
