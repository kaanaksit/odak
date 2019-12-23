import plyfile
from plyfile import PlyData, PlyElement
from odak import np

def read_PLY(fn):
    """
    Definition to read a PLY file and extract meshes from a given PLY file.

    Parameters
    ----------
    fn          : string
                  Filename of a PLY file.

    Returns
    ----------
    mesh        : ndarray
                  Meshes from a given PLY file.
    """
    with open(fn,'rb') as f:
        plydata = PlyData.read(f)
    mesh = plydata.elements[0].data
    return mesh

