import odak # (1)
import torch
import sys
from odak.learn.perception.color_conversion import display_color_hvs
from odak.learn.tools import load_image, save_image, resize


def test():
    torch.manual_seed(0)
    the_number_of_primaries = 3
    target_primaries = torch.rand(1,
                                  the_number_of_primaries,
                                  1024,
                                  1024
                                  ) # (2)

    multi_spectrum = torch.rand(the_number_of_primaries,
                                301
                                ) # (3)
    device_ = torch.device('cpu') # (4)
    display_color = display_color_hvs(read_spectrum ='tensor',
                                      primaries_spectrum=multi_spectrum,
                                      device = device_)
    lms_color = display_color.primaries_to_lms(target_primaries) # (5)
    third_stage = display_color.second_to_third_stage(display_color.primaries_to_lms(target_primaries))
    sample_input_image = torch.rand(1, 3, 100, 100)
    sample_ground_truth = torch.rand(1, 3, 100, 100)
    sample_loss = display_color(sample_input_image, sample_ground_truth)
    assert True == True

if __name__ == "__main__":
    sys.exit(test())
