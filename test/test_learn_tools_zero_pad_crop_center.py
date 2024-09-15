import sys
import odak
import torch


def test(output_directory = 'test_output'):
    odak.tools.check_directory(output_directory)
    image0 = torch.zeros(1, 3, 50, 50)
    image0[:, :, ::2, :: 2] = 1.
    image0_zero_padded = odak.learn.tools.zero_pad(image0)
    odak.learn.tools.save_image('{}/image0_padded.png'.format(output_directory), image0_zero_padded, cmin = 0., cmax = 1.)

    image1 = torch.zeros(1, 50, 50, 3)
    image1[:, ::3, ::3, :] = 1.
    image1_zero_padded = odak.learn.tools.zero_pad(image1)
    odak.learn.tools.save_image('{}/image1_padded.png'.format(output_directory), image1_zero_padded, cmin = 0., cmax = 1.)


    image0_cropped = odak.learn.tools.crop_center(image0_zero_padded)
    odak.learn.tools.save_image('{}/image0_cropped.png'.format(output_directory), image0_cropped, cmin = 0., cmax = 1.)
    image1_cropped = odak.learn.tools.crop_center(image1_zero_padded)
    odak.learn.tools.save_image('{}/image1_padded.png'.format(output_directory), image1_cropped, cmin = 0., cmax = 1.)

    assert True == True


if __name__ == '__main__':
    sys.exit(test())
