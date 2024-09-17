import torch
import sys
import odak.learn.tools.loss as loss


def test():
    # test residual block
    image = [torch.randn(1, 3, 32, 32), torch.randn(1, 32, 32), torch.randn(32, 32), torch.randn(3, 32, 32)]
    ground_truth = [torch.randn(1, 3, 32, 32), torch.randn(1, 32, 32), torch.randn(32, 32), torch.randn(3, 32, 32)]
    for idx, (img, pred) in enumerate(zip(image, ground_truth)):
        print(f'Running test {idx}, input shape: {img.size()}...')
        y = loss.multi_scale_total_variation_loss(img, levels = 4)
        y = loss.total_variation_loss(img)
        y = loss.histogram_loss(img, pred, bins = 16, limits = [0., 1.])
        roi_high = [0, 16, 0, 16]
        roi_low = [16, 32, 16, 32]
        y = loss.weber_contrast(img, roi_high, roi_low)
        y = loss.michelson_contrast(img, roi_high, roi_low)
        y = loss.wrapped_mean_squared_error(img, pred, reduction='sum')
    value = torch.tensor(1.0, dtype=torch.float)
    y = loss.radial_basis_function(value = value, epsilon = 0.5)
    assert True == True

if __name__ == '__main__':
    sys.exit(test())
