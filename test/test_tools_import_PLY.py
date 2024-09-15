import odak.raytracing as raytracer
import sys
import odak
from odak.tools.asset import read_PLY, write_PLY


def test(output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    triangles = read_PLY('./test/data/sample.ply')
    centers = raytracer.center_of_triangle(triangles)
    normals = raytracer.get_triangle_normal(triangles, centers)
    write_PLY(triangles, savefn = '{}/output.ply'.format(output_directory))
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
