import odak
import torch
import sys


def test():
    device = torch.device('cpu')
    input_rgb_image = torch.randn((1, 3, 256, 256)).to(device)

    ycrcb_image = odak.learn.perception.color_conversion.rgb_2_ycrcb(input_rgb_image)
    rgb_image = odak.learn.perception.color_conversion.ycrcb_2_rgb(ycrcb_image)

    linear_rgb_image = odak.learn.perception.color_conversion.rgb_to_linear_rgb(rgb_image)
    rgb_image = odak.learn.perception.color_conversion.linear_rgb_to_rgb(linear_rgb_image)
    
    xyz_image = odak.learn.perception.color_conversion.linear_rgb_to_xyz(linear_rgb_image)
    linear_rgb_image = odak.learn.perception.color_conversion.xyz_to_linear_rgb(xyz_image)

    hsv_image = odak.learn.perception.color_conversion.rgb_to_hsv(rgb_image)
    rgb_image = odak.learn.perception.color_conversion.hsv_to_rgb(hsv_image)

    lms_image = odak.learn.perception.color_conversion.rgb_to_lms(rgb_image)
    rgb_image = odak.learn.perception.color_conversion.lms_to_rgb(lms_image)    
    hvs_second_stage_image = odak.learn.perception.color_conversion.lms_to_hvs_second_stage(lms_image)
    
    input_srgb_image = torch.randn((3, 256, 256)).to(device)
    srgb_image = odak.learn.perception.color_conversion.srgb_to_lab(input_srgb_image)
    lab_image = odak.learn.perception.color_conversion.lab_to_srgb(srgb_image)


    device_ = torch.device('cpu')
    display_color = odak.learn.perception.color_conversion.display_color_hvs(read_spectrum ='precomputed',
                                    spectrum_data_root = './backlight/',
                                    device=device_)
    input_rgb_image = torch.rand(1, 1024, 1024, 3)
    lms_image = display_color.rgb_to_lms(input_rgb_image)

    input_srgb_image = torch.randn((3, 256, 256)).to(device_)
    target_srgb_image = torch.randn((3, 256, 256)).to(device_)
    mapped_srgb_image = odak.learn.perception.color_conversion.color_map(input_srgb_image, target_srgb_image, model = 'Lab Stats')

    assert True == True


if __name__ == "__main__":
    sys.exit(test())


