import odak.learn
import torch
import sys


def test_learned_perceptual_losses():
    input_tensor_dims = [1, 3, 512, 512]
    test_image = torch.ones(input_tensor_dims) * 0.5 + torch.randn(input_tensor_dims) * 0.1
    test_image = torch.clamp(test_image, 0.0, 1.0)
    test_target = torch.ones(input_tensor_dims) * 0.5

    cvvdp = odak.learn.perception.learned_perceptual_losses.CVVDP()
    fvvdp = odak.learn.perception.learned_perceptual_losses.FVVDP()
    lpips = odak.learn.perception.learned_perceptual_losses.LPIPS()

    l_CVVDP = cvvdp(test_image, test_target, dim_order = 'BCHW')
    l_FVVDP = fvvdp(test_image, test_target, dim_order = 'BCHW') 
    l_LPIPS = lpips(test_image, test_target)

    assert True == True

if __name__ == "__main__":
    sys.exit(test_learned_perceptual_losses())
