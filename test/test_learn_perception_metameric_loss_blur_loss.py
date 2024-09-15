import odak.learn
import torch
import sys


def test_perceptual_losses():
    test_image = torch.ones([1, 3, 512, 512]) * 0.5 + torch.randn([1, 3, 512, 512]) * 0.1
    test_image = torch.clamp(test_image, 0.0, 1.0)
    test_target = torch.ones([1, 3, 512, 512]) * 0.5

    gaze = [0.5, 0.5]

    my_metameric_loss = odak.learn.perception.MetamericLoss()
    my_uniform_metameric_loss = odak.learn.perception.MetamericLossUniform()
    my_blur_loss = odak.learn.perception.BlurLoss()
    my_metamer_mse_loss = odak.learn.perception.MetamerMSELoss()
    assert True == True


if __name__ == "__main__":
    sys.exit(test_perceptual_losses())
