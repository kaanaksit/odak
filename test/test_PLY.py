import odak.raytracing as raytracer
import sys
from odak.tools.asset import read_PLY

def test():
    mesh = read_PLY('sample.ply')
    import numpy as np
    mesh = np.array(mesh)
    mesh = mesh.reshape((100,100,3))
    print(mesh.shape)
    assert True==True

if __name__ == '__main__':
    sys.exit(test())
