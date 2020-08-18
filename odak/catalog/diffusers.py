from odak import np
import odak.catalog
from odak.raytracing.primitives import define_plane,bring_plane_to_origin
from odak.raytracing.boundary import intersect_w_surface
from odak.wave.parameters import wavenumber

class thin_diffuser():
    """
    A class to represent a thin diffuser. This is generally useful for raytracing and wave calculations.
    """
    def __init__(self,phase=None,shape=[10.,10.],center=[0.,0.,0.],angles=[0.,0.,0.],diffusion_angles=[5.,5.],diffusion_no=[3,3]):
        """
        Class to represent a simple planar detector.

        Parameters
        ----------
        phase            : ndarray
                           Initial phase to be loaded. If non provided, it will start with a random phase.
        shape            : list
                           Shape of the detector.
        center           : list
                           Center of the detector.
        angles           : list
                           Rotation angles of the detector.
        diffusion angles : list
                           Full angle of diffusion along two axes.
        diffusion_no     : list
                           Number of rays to be generated along two axes at each diffusion.
        """
        self.settings           = {
                                   'center'                   : center,
                                   'angles'                   : angles,
                                   'rotation mode'            : 'XYZ',
                                   'shape'                    : shape,
                                   'diffusion angles'         : diffusion_angles,
                                   'number of diffusion rays' : diffusion_no
                                  }
        self.plane              = define_plane(
                                               self.settings['center'],
                                               angles=self.settings['angles']
                                              )
        self.k                  = wavenumber(0.05)
        self.diffusion_rays     = np.mgrid[
                                           0:self.settings["number of diffusion rays"][0],
                                           0:self.settings["number of diffusion rays"][1]
                                          ].astype(np.float32)
        self.diffusion_rays[0]  = self.diffusion_rays[0] - np.amax(self.diffusion_rays[0])/2.
        self.diffusion_rays[1]  = self.diffusion_rays[1] - np.amax(self.diffusion_rays[1])/2.
        self.diffusion_rays[0]  = self.diffusion_rays[0]/np.amax(self.diffusion_rays[0])*self.settings["diffusion angles"][0]/2.
        self.diffusion_rays[1]  = self.diffusion_rays[1]/np.amax(self.diffusion_rays[1])*self.settings["diffusion angles"][1]/2.
        self.diffusion_rays     = np.array(
                                           [
                                            self.diffusion_rays[0].flatten(),
                                            self.diffusion_rays[1].flatten(),
                                            self.diffusion_rays[1].flatten()
                                           ]
                                          )
        self.diffusion_rays     = self.diffusion_rays.reshape(self.diffusion_rays.shape[1],self.diffusion_rays.shape[0])
        self.diffusion_rays[:,2] = 1
        print(self.diffusion_rays)
        import sys;sys.exit()

    def raytrace(self,ray):
        """
        """
        if len(ray.shape) == 2:
            ray = ray.reshape((1,ray.shape[0],ray.shape[1]))
        normal,distance        = intersect_w_surface(ray,self.plane)
        ray_diffuser           = np.copy(ray)
        ray_diffuser[:,0,:]    = normal[:,0,:]
        print(ray_diffuser.shape)
        ray_diffuser           = np.repeat(
                                           ray_diffuser,
                                           self.settings['number of diffusion rays'][0]*self.settings['number of diffusion rays'][1],
                                           axis=0
                                          )
        print(ray_diffuser[:,0])
        import sys;sys.exit()
         
        new_field              = np.zeros(self.phase.shape)
        if type(k) == type(None):
            k = self.k
        new_field[:,:,channel] = wave.propagate_plane_waves(field,distance,k)*self.phase
        return new_ray,new_field
