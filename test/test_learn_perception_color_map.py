import odak
import torch
import sys
import odak.learn.perception.color_conversion as color_map


def test():
    device = torch.device('cpu')
    input_srgb_image = torch.randn((3, 256, 256)).to(device)
    target_srgb_image = torch.randn((3, 256, 256)).to(device)
    mapped_srgb_image = color_map.color_map(input_srgb_image, target_srgb_image, model = 'Lab Stats')
    assert True == True


if __name__ == "__main__":
    sys.exit(test())
