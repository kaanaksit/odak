import odak
import sys
import numpy as np

def test():
    test = np.random.rand(100, 100)
    odak.tools.save_image('image_8_bit_monochrome.png', test, color_depth=8, cmin=0., cmax=1.)
    odak.tools.save_image('image_16_bit_monochrome.png', test, color_depth=16, cmin=0, cmax=1.)
    test = np.random.rand(100, 100, 3)
    odak.tools.save_image('image_8_bit_color.png', test, color_depth=8, cmin=0., cmax=1.)
    odak.tools.save_image('image_16_bit_color.png', test, color_depth=16, cmin=0, cmax=1.)
    assert True == True

if __name__ == '__main__':
    sys.exit(test())
