import sys
import os
import odak
import numpy as np


def test():
    n = [100, 100]
    ranges = [[500., 3000.], [10., 100.]]
    x, y = np.mgrid[0:n[0], 0:n[1]]
    focals = x*(ranges[0][1]-ranges[0][0])/n[0]+ranges[0][0]
    apertures = y*(ranges[1][1]-ranges[1][0])/n[1]+ranges[1][0]
    wavelength = 0.0005
    resolutions = np.zeros((n[0], n[1], 3))
    for i in range(0, n[0]):
        for j in range(0, n[1]):
            resolutions[i, j, 0] = focals[i, j]
            resolutions[i, j, 1] = apertures[i, j]
            resolutions[i, j, 2] = odak.wave.rayleigh_resolution(
                diameter=resolutions[i, j, 1],
                focal=resolutions[i, j, 0],
                wavelength=wavelength
            )*1000  # Conversion to um.
#    figure = odak.visualize.surfaceshow(
#        title='Spatial resolution',
#        labels=[
#            'Throw distance (mm)',
#            'Aperture size (mm) ',
#            'Spatial resolution (um)'
#        ],
#        types=[
#            'log',
#            'log',
#            'log'
#        ],
#        font_size=16,
#        tick_no=[2, 2, 4],
#    )
#    figure.add_surface(
#        data_x=resolutions[:, :, 0],
#        data_y=resolutions[:, :, 1],
#        data_z=resolutions[:, :, 2],
#        contour=False
#    )
#    figure.show()
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
