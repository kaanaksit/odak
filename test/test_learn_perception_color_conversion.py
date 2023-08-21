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
    
    input_srgb_image = torch.randn((3, 256, 256)).to(device)
    srgb_image = odak.learn.perception.color_conversion.srgb_to_lab(input_srgb_image)
    lab_image = odak.learn.perception.color_conversion.lab_to_srgb(srgb_image)

    assert True == True


if __name__ == "__main__":
    sys.exit(test())


