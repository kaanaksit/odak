import sys
import odak
import torch
import odak.learn.tools.loss as loss


def test(header="test/test_learn_tools_losses.py"):
    image = [
        torch.randn(1, 3, 32, 32),
        torch.randn(1, 32, 32),
        torch.randn(32, 32),
        torch.randn(3, 32, 32),
    ]
    ground_truth = [
        torch.randn(1, 3, 32, 32),
        torch.randn(1, 32, 32),
        torch.randn(32, 32),
        torch.randn(3, 32, 32),
    ]
    for idx, (img, pred) in enumerate(zip(image, ground_truth)):
        odak.log.logger.info(
            f"{header} -> Running test {idx}, input shape: {img.size()}..."
        )
        loss.multi_scale_total_variation_loss(img, levels=4)
        loss.total_variation_loss(img)
        loss.histogram_loss(img, pred, bins=16, limits=[0.0, 1.0])
        loss.wrapped_mean_squared_error(img, pred, reduction="sum")
    value = torch.tensor(1.0, dtype=torch.float)
    loss.radial_basis_function(value=value, epsilon=0.5)
    assert True


if __name__ == "__main__":
    sys.exit(test())
