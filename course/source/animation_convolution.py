import odak
import torch
import sys


def main():
    filename_image = '../media/10591010993_80c7cb37a6_c.jpg'
    image = odak.learn.tools.load_image(filename_image, normalizeby = 255., torch_style = True)[0:3].unsqueeze(0)
    kernel = odak.learn.tools.generate_2d_gaussian(kernel_length = [12, 12], nsigma = [21, 21])
    kernel = kernel / kernel.max()
    result = torch.zeros_like(image)
    result = odak.learn.tools.zero_pad(result, size = [image.shape[-2] + kernel.shape[0], image.shape[-1] + kernel.shape[1]])
    step = 0
    for i in range(image.shape[-2]):
        for j in range(image.shape[-1]):
            for ch in range(image.shape[-3]):
                element = image[:, ch, i, j]
                add = kernel * element
                result[:, ch, i : i + kernel.shape[0], j : j + kernel.shape[1]] += add
            if (i * image.shape[-1] + j) % 1e4 == 0:
                filename = 'step_{:04d}.png'.format(step)
                odak.learn.tools.save_image( filename, result, cmin = 0., cmax = 100.)
                step += 1
    cmd = ['convert', '-delay', '1', '-loop', '0', '*.png', '../media/convolution_animation.gif']
    odak.tools.shell_command(cmd)
    cmd = ['rm', '*.png']
    odak.tools.shell_command(cmd)


if __name__ == '__main__':
    sys.exit(main())
