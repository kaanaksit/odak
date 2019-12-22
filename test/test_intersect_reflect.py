import odak.raytracing as raytracer
import sys

def detector_to_light_source(detector_location,triangle,light_source):
    """
    This definition describe an optical path from a given pinhole detector to a given triangle then from that triangle to a light source. The definition returns False if the path isn't feasible.

    Parameters
    ----------
    detector_location     : list
                            Pinhole detector location in X,Y,Z.
    triangle              : list
                            Points that described a triangle in X,Y,Z.
    light_source          : list
                            Light source described in packed single variable form (see odak.raytracing.define_circle).

    Returns
    ----------
    opl                   : float
                            Total optical path length.
    """
    center_of_triangle        = raytracer.center_of_triangle(triangle)
    ray_detector_triangle     = raytracer.create_ray_from_two_points(
                                                                     detector_location,
                                                                     center_of_triangle
                                                                    )
    normal_triangle,d_det_tri = raytracer.intersect_w_triangle(
                                                               ray_detector_triangle,
                                                               triangle
                                                              )
    ray_reflection            = raytracer.reflect(
                                                  ray_detector_triangle,
                                                  normal_triangle
                                                 )
    normal_source,d_tri_sou   = raytracer.intersect_w_circle(
                                                             ray_reflection,
                                                             light_source
                                                            )
    if type(normal_source) == type(False):
        return False
    opl                       = d_det_tri + d_tri_sou
    return opl

def test():
    detector_location = [2.,0.,0.]
    triangle          = [
                         [ 10., 10., 10.],
                         [  0., 10., 10.],
                         [  0.,  0., 10.]
                        ]
    circle_center     = [0.,0.,0.]
    circle_angles     = [0.,0.,0.]
    circle_radius     = 15.
    circle = raytracer.define_circle(
                                     angles=circle_angles,
                                     center=circle_center,
                                     radius=circle_radius
                                    )

    opl    = detector_to_light_source(
                                      detector_location,
                                      triangle,
                                      circle
                                     )
    assert True==True

if __name__ == '__main__':
    sys.exit(test())
