import odak.raytracing as raytracer
import odak.catalog as catalog
import sys
from odak.tools.sample import grid_sample, batch_of_rays

def test():
    sample_entry_points = grid_sample(
        no=[2, 2],
        size=[100., 100.],
        center=[0., 0., 0.],
        angles=[0., 0., 0.]
    )
    sample_exit_points = grid_sample(
        no=[2, 2],
        size=[100., 100.],
        center=[0., 50., 100.],
        angles=[0., 0., 0.]
    )
    rays = raytracer.create_ray_from_two_points(
        sample_entry_points,
        sample_exit_points
    )
    diffuser = catalog.diffusers.thin_diffuser(
        shape=[50, 50],
        center=[0., 0., 100.],
        angles=[0., 0., 0.],
        diffusion_no=[5, 2],
        diffusion_angle=10.
    )
    new_rays, normal, distance = diffuser.raytrace(rays)
    detector = catalog.detectors.plane_detector(
        resolution=[5, 5],
        shape=[50, 50],
        center=[0., 0., 200.],
        angles=[0., 0., 0.],
    )
    normals, distances = detector.raytrace(new_rays)
#    import odak
#    ray_visualize = odak.visualize.rayshow(line_width=5)
#    ray_visualize.add_line(rays[:,0],normal[:,0])
#    ray_visualize.add_line(new_rays[:,0],normals[:,0])
#    ray_visualize.show()
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
