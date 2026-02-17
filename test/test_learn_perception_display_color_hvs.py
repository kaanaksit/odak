import odak  # (1)
import torch
import sys
from odak.learn.perception.color_conversion import display_color_hvs

header = "test/test_learn_perception_display_color_hvs.py"


def test(device=torch.device("cpu"), output_directory="test_output"):
    odak.tools.check_directory(output_directory)
    torch.manual_seed(0)

    image_rgb = (
        odak.learn.tools.load_image(
            "test/data/fruit_lady.png", normalizeby=255.0, torch_style=True
        )
        .unsqueeze(0)
        .to(device)
    )  # (2)

    the_number_of_primaries = 3
    multi_spectrum = torch.zeros(the_number_of_primaries, 301)  # (3)
    multi_spectrum[0, 200:250] = 1.0
    multi_spectrum[1, 130:145] = 1.0
    multi_spectrum[2, 0:50] = 1.0

    display_color = display_color_hvs(
        read_spectrum="tensor", primaries_spectrum=multi_spectrum, device=device
    )  # (4)

    image_lms_second_stage = display_color.primaries_to_lms(image_rgb)  # (5)
    image_lms_third_stage = display_color.second_to_third_stage(
        image_lms_second_stage
    )  # (6)

    odak.learn.tools.save_image(
        "{}/image_rgb.png".format(output_directory),
        image_rgb,
        cmin=0.0,
        cmax=image_rgb.max(),
    )

    odak.learn.tools.save_image(
        "{}/image_lms_second_stage.png".format(output_directory),
        image_lms_second_stage,
        cmin=0.0,
        cmax=image_lms_second_stage.max(),
    )

    odak.learn.tools.save_image(
        "{}/image_lms_third_stage.png".format(output_directory),
        image_lms_third_stage,
        cmin=0.0,
        cmax=image_lms_third_stage.max(),
    )

    image_rgb_noisy = image_rgb * 0.6 + torch.rand_like(image_rgb) * 0.4  # (7)
    loss_lms = display_color(image_rgb, image_rgb_noisy)  # (8)
    odak.log.logger.info(
        "{} -> The third stage LMS sensation difference between two input images is {:.10f}.".format(
            header, loss_lms
        )
    )
    assert True == True


if __name__ == "__main__":
    sys.exit(test())
