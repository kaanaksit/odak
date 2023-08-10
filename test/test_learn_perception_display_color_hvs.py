import odak
import torch
import sys
from odak.learn.perception.color_conversion import display_color_hvs

def test():
    device_ = torch.device('cpu')
    display_color = display_color_hvs(read_spectrum ='precomputed',
                                    spectrum_data_root = './backlight/',
                                    device=device_)
    input_rgb_image = torch.rand(1, 1024, 1024, 3)
    lms_image = display_color.rgb_to_lms(input_rgb_image)
    assert True == True

if __name__ == "__main__":
    sys.exit(test())
