import odak.learn
import torch
import sys


def test_perceptual_losses():
    test_image = (
        torch.ones([1, 3, 512, 512]) * 0.5 + torch.randn([1, 3, 512, 512]) * 0.1
    )
    test_image = torch.clamp(test_image, 0.0, 1.0)
    torch.ones([1, 3, 512, 512]) * 0.5


    odak.learn.perception.MetamericLoss()
    odak.learn.perception.MetamericLossUniform()
    odak.learn.perception.BlurLoss()
    odak.learn.perception.MetamerMSELoss()
    assert True


if __name__ == "__main__":
    sys.exit(test_perceptual_losses())
