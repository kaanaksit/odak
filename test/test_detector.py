import odak.raytracing as raytracer
import odak.catalog as catalog
import sys
from odak.tools.sample import grid_sample, batch_of_rays


def test():
    sample_entry_points = grid_sample(
        no=[4, 4],
        size=[100., 100.],
        center=[0., 0., 0.],
        angles=[0., 0., 0.]
    )
    sample_exit_points = grid_sample(
        no=[4, 4],
        size=[100., 100.],
        center=[0., 0., 100.],
        angles=[0., 0., 0.]
    )
    rays = raytracer.create_ray_from_two_points(
        sample_entry_points,
        sample_exit_points
    )
    detector = catalog.detectors.plane_detector(
        resolution=[5, 5],
        shape=[50, 50],
        center=[0., 0., 200.],
        angles=[0., 10., 0.]
    )
    normals, distances = detector.raytrace(rays)
    print(detector.get_field())
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
