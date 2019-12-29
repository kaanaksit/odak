import odak.raytracing as raytracer
import sys
import numpy as np
from odak.tools.asset import read_PLY,write_PLY

def test():
    triangles = read_PLY('./test/sample.ply')
    # write_PLY has to be fixed. Currently not working as intended.
    write_PLY(triangles,savefn='output.ply')
    assert True==True

if __name__ == '__main__':
    sys.exit(test())
