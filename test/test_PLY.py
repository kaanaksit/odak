import odak.raytracing as raytracer
import sys
from odak.tools.asset import read_PLY

def test():
    mesh = read_PLY('sample.ply')
    assert True==True

if __name__ == '__main__':
    sys.exit(test())
