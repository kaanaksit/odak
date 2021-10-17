from test.test_PLY import test
import odak.learn
import torch

import sys


def test_perceptual_losses():
    # Make a simple test image with noise, and a target image without noise.
    test_image = torch.ones([1, 3, 512, 512])*0.5 + \
        torch.randn([1, 3, 512, 512])*0.1
    test_image = torch.clamp(test_image, 0.0, 1.0)
    test_target = torch.ones([1, 3, 512, 512])*0.5

    # Specify gaze location in normalized image coordinates (in range [0, 1]).
    gaze = [0.5, 0.5]

    # Create the 3 loss functions.
    my_metameric_loss = odak.learn.perception.MetamericLoss()
    my_uniform_metameric_loss = odak.learn.perception.MetamericLossUniform()
    my_blur_loss = odak.learn.perception.BlurLoss()
    my_metamer_mse_loss = odak.learn.perception.MetamerMSELoss()

    # Measure and print the 3 losses.
    print("Metameric Loss:", my_metameric_loss(
        test_image, test_target, gaze=gaze).item())
    print("Metameric Loss:", my_uniform_metameric_loss(
        test_image, test_target).item())
    print("Metamer MSE Loss:", my_metamer_mse_loss(
        test_image, test_target, gaze=gaze).item())
    print("Blur Loss:", my_blur_loss(test_image, test_target, gaze=gaze).item())

    return True


if __name__ == "__main__":
    sys.exit(test_perceptual_losses())
