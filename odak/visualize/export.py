import torch
import numpy as np
from plyfile import PlyData, PlyElement


class PLY_object():
    """
    A class to create an PLY file. This is useful for visualizing rays in different CAD software such as FreeCAD, Blender or Meshlab.
    """

    def __init__(self):
        self.pnts = None
        self.tris = None

    def draw_a_ray(self, point0, point1, k=0.02, color=[255, 255, 255]):
        """
        Definition to draw a ray.

        Parameters
        ----------
        point0      : list
                      List that contains X, Y, and Z start locations of a ray.
        point1      : list
                      List that contains X, Y, and Z end locations of a ray.
        k           : float
                      Thickness of a ray.
        color       : list
                      List that contains red, green and blue color channel values (8 bit).
        """
        if torch.is_tensor(point0):
            point0 = point0.cpu().detach().numpy()
        if torch.is_tensor(point1):
            point1 = point1.cpu().detach().numpy()
        import numpy as np_cpu
        point0 = np_cpu.reshape(np_cpu.array(point0), (3))
        point1 = np_cpu.reshape(np_cpu.array(point1), (3))
        if type(self.pnts) == type(None):
            a = 0
        elif type(self.pnts) != type(None):
            a = np_cpu.asarray(self.pnts).shape[0]
        pnts = np_cpu.array([
            (point0[0], point0[1], point0[2]),
            (point0[0]+k, point0[1], point0[2]),
            (point0[0]+k, point0[1]+k, point0[2]),
            (point0[0], point0[1]+k, point0[2]),
            (point1[0], point1[1], point1[2]),
            (point1[0]+k, point1[1], point1[2]),
            (point1[0]+k, point1[1]+k, point1[2]),
            (point1[0], point1[1]+k, point1[2])],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        )
        tris = np_cpu.array([
            ([0+a, 3+a, 1+a], color[0], color[1], color[2]),
            ([1+a, 3+a, 2+a], color[0], color[1], color[2]),
            ([0+a, 4+a, 7+a], color[0], color[1], color[2]),
            ([0+a, 7+a, 3+a], color[0], color[1], color[2]),
            ([4+a, 5+a, 6+a], color[0], color[1], color[2]),
            ([4+a, 6+a, 7+a], color[0], color[1], color[2]),
            ([5+a, 1+a, 2+a], color[0], color[1], color[2]),
            ([5+a, 2+a, 6+a], color[0], color[1], color[2]),
            ([2+a, 3+a, 6+a], color[0], color[1], color[2]),
            ([3+a, 7+a, 6+a], color[0], color[1], color[2]),
            ([0+a, 1+a, 5+a], color[0], color[1], color[2]),
            ([0+a, 5+a, 4+a], color[0], color[1], color[2])],
            dtype=[('vertex_indices', 'i4', (3,)),
                   ('red', 'u1'), ('green', 'u1'),
                   ('blue', 'u1')]
        )
        if type(self.pnts) != type(None):
            self.tris = np_cpu.concatenate([np.copy(self.tris), np.copy(tris)])
            self.pnts = np_cpu.concatenate([np.copy(self.pnts), np.copy(pnts)])
        else:
            self.tris = np_cpu.copy(tris)
            self.pnts = np_cpu.copy(pnts)

    def save_PLY(self, savefn='out.ply'):
        """
        Definition to save the PLY file.

        Parameters
        ----------
        savefn      : str
                      Filename with a complete path to save the PLY file.

        """
        el1 = PlyElement.describe(
            self.pnts, 'vertex', comments=['Vertex data'])
        el2 = PlyElement.describe(self.tris, 'face', comments=['Face data'])
        PlyData([el1, el2], text="True").write(savefn)
