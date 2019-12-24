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
    triangles   : ndarray
                  Triangles from a given PLY file. Note that the triangles coming out of this function isn't always structured in the right order and with the size of (MxN)x3. You can use numpy's reshape to restructure it to mxnx3 if you know what you are doing.
    """
    with open(fn,'rb') as f:
        plydata = PlyData.read(f)
    triangle_ids = np.vstack(plydata['face'].data['vertex_indices'])
    triangles    = []
    for vertex_ids in triangle_ids:
        triangle = [
                    plydata['vertex'][vertex_ids[0]].tolist(),
                    plydata['vertex'][vertex_ids[1]].tolist(),
                    plydata['vertex'][vertex_ids[2]].tolist()
                   ]
        triangles.append(triangle)
    triangles = np.array(triangles)
    return triangles

def write_PLY(triangles,savefn='output.ply'):
    """
    Definition to generate a PLY file from given points.

    Parameters
    ----------
    triangles   : ndarray
                  List of triangles with the size of Mx3x3.
    savefn      : string
                  Filename for a PLY file.
    """
    tris  = []
    pnts  = []
    color = [255,255,255]
    for tri_id in range(triangles.shape[0]):
       tris.append(
                   (
                    [3*tri_id,3*tri_id+1,3*tri_id+2], 
                    color[0],
                    color[1],
                    color[2]
                   )
                  )
       for i in range(0,3):
           pnts.append(
                       (
                        triangles[tri_id][i][0],
                        triangles[tri_id][i][1],
                        triangles[tri_id][i][2]
                       )
                      )
    tris   = np.asarray(tris, dtype=[('vertex_indices', 'i4', (3,)),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    pnts   = np.asarray(pnts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    # Save mesh.
    el1       = PlyElement.describe(pnts, 'vertex', comments=['Vertex data'])
    el2       = PlyElement.describe(tris, 'face', comments=['Face data'])
    PlyData([el1,el2],text="True").write(savefn)

def write_PLY_from_points(points,savefn='output.ply'):
    """
    Definition to generate a PLY file from given points.

    Parameters
    ----------
    points      : ndarray
                  List of points with the size of MxNx3.
    savefn      : string
                  Filename for a PLY file.

    """
    # Generate equation
    roi     = np.zeros((2,points.shape[0],points.shape[1]))
    roi[0]  = np.copy(points[:,:,0])
    roi[1]  = np.copy(points[:,:,1])
    zz      = np.copy(points[:,:,2])
    samples = [points.shape[0],points.shape[1]]
    # Generate vertices.
    pnts   = []
    tris   = []
    for idx in range(0,samples[0]):
        for idy in range(0,samples[1]):
            pnt  = (roi[0][idx][idy]    , roi[1][idx][idy]    , zz[idx][idy])
            pnts.append(pnt)
    m = samples[0]*samples[1]
    color = [255,255,255]
    for idx in range(0,samples[0]-1):
        for idy in range(0,samples[1]-1):
            tris.append(([idy+(idx+1)*samples[0], idy+idx*samples[0]  , idy+1+idx*samples[0]], color[0], color[1], color[2]))
            tris.append(([idy+(idx+1)*samples[0], idy+1+idx*samples[0], idy+1+(idx+1)*samples[0]], color[0], color[1], color[2]))
    tris   = np.asarray(tris, dtype=[('vertex_indices', 'i4', (3,)),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    pnts   = np.asarray(pnts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    # Save mesh.
    el1       = PlyElement.describe(pnts, 'vertex', comments=['Vertex data'])
    el2       = PlyElement.describe(tris, 'face', comments=['Face data'])
    PlyData([el1,el2],text="True").write(savefn)

