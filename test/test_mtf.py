import sys
import odak
import numpy as np


def test():
    px_size = 30./151.
    dist = 580.  # 58 cm
    angsize = np.degrees(np.arctan(px_size/dist))
    image = np.random.rand(2000, 2000)
    location = [340, 360, 1010, 1050]
    line_x, line_y, roi = odak.measurement.roi(
        image,
        location=location,
    )
    mtf, freq = odak.measurement.modulation_transfer_function(
        line_x,
        line_y,
        px_size=[angsize, angsize],
    )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
