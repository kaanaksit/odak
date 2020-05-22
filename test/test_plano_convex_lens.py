#!/usr/bin/env python

import sys

def test_plano_convex():
    import odak.raytracing as raytracer
    import odak.tools as tools
    import odak.catalog as catalog
    end_points        = tools.grid_sample(
                                          no=[5,5],
                                          size=[2.0,2.0],
                                          center=[0.,0.,0.],
                                          angles=[0.,0.,0.]
                                         )
    start_point       = [0.,0.,-5.]
    rays              = raytracer.create_ray_from_two_points(
                                                             start_point,
                                                             end_points
                                                            )
    lens              = catalog.plano_convex_lens()
    normals,distances = lens.intersect(rays)
    return True

    from odak import np
    import plotly
    import plotly.graph_objs as go    
    if np.__name__ == 'cupy':
        df = np.asnumpy(normals[:,0])
        dx = np.asnumpy(end_points)
        dy = np.asnumpy(start_point)
    else:
        df = normals[:,0]
        dx = end_points
        dy = start_point
    trace0 = go.Scatter3d(x=df[:,0],y=df[:,1],z=df[:,2])
    trace1 = go.Scatter3d(x=dx[:,0],y=dx[:,1],z=dx[:,2])
    trace2 = go.Scatter3d(x=[dy[0],],y=[dy[1],],z=[dy[2],])
    data   = [trace0,trace1,trace2]
    fig    = dict(data=data)
    plotly.offline.plot(fig)
    assert True==True

if __name__ == '__main__':
    sys.exit(test_plano_convex())
