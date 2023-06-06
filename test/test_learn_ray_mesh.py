#!/usr/bin/env python


import sys
import odak
import torch
from tqdm import tqdm


def test():
    triangles = odak.learn.raytracing.define_plane_mesh()

    visualize = True
    if visualize:
        ray_diagram = odak.visualize.plotly.rayshow(
                                                    columns = 2,
                                                    line_width = 3.,
                                                    marker_size = 3.,
                                                    subplot_titles = ['Before optimization', 'After optimization']
                                                   ) 
        for triangle_id in range(triangles.shape[0]):
            ray_diagram.add_triangle(triangles[triangle_id], column = 1, color = 'black')
        html = ray_diagram.save_offline()
#        print(html)
    assert True == True
      

if __name__ == '__main__':
    sys.exit(test())
