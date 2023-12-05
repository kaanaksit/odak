import odak.raytracing as raytracer
import sys
from odak.tools.asset import read_PLY, write_PLY


def test():
    triangles = read_PLY('./test/sample.ply')
    centers = raytracer.center_of_triangle(triangles)
    normals = raytracer.get_triangle_normal(triangles, centers)
    write_PLY(triangles, savefn='output.ply')
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
