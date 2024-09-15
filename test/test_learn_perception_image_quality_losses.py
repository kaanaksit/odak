import odak.learn
import torch
import sys


def test_image_quality_losses():
    input_tensor_dims = [1, 3, 512, 512]
    test_image = torch.ones(input_tensor_dims) * 0.5 + torch.randn(input_tensor_dims) * 0.1
    test_image = torch.clamp(test_image, 0.0, 1.0)
    test_target = torch.ones(input_tensor_dims) * 0.5

    psnr = odak.learn.perception.image_quality_losses.PSNR()
    ssim = odak.learn.perception.image_quality_losses.SSIM()
    msssim = odak.learn.perception.image_quality_losses.MSSSIM()

    l_PSNR = psnr(test_image, test_target)
    l_SSIM = ssim(test_image, test_target)
    l_MSSSIM = msssim(test_image, test_target)

    assert True == True


if __name__ == "__main__":
    sys.exit(test_image_quality_losses())
