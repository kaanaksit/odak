import odak
import torch
import sys
from odak.learn.perception.color_conversion import display_color_hvs
from odak.learn.tools import load_image, save_image, resize


def test():
    target_image = torch.rand(10, 3, 1024, 1024)
    rgb_spectrum = torch.rand(3, 301) # 400-700 nm
    device_ = torch.device('cpu')
    display_color = display_color_hvs(read_spectrum ='default',
                                      primaries_spectrum=rgb_spectrum,
                                      device = device_)
    lms_image = display_color.rgb_to_lms(target_image)

    assert True == True

if __name__ == "__main__":
    sys.exit(test())
