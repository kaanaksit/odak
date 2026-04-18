import sys
import os
import torch
import numpy as np
import odak
from odak.learn.wave import decompose_double_phase, compose_double_phase


def create_test_tensor(shape):
    """
    Create a tensor where each 2x2 patch has:
    [0,0] = 1, [0,1] = 0
    [1,0] = 0, [1,1] = 1
    """
    h, w = shape
    tensor = torch.zeros(h, w)

    # [0,0] positions (even rows, even cols) -> 1
    tensor[0::2, 0::2] = 1.0
    # [1,1] positions (odd rows, odd cols) -> 1
    tensor[1::2, 1::2] = 1.0
    # [0,1] positions (even rows, odd cols) -> 0 (already zero)
    # [1,0] positions (odd rows, even cols) -> 0 (already zero)

    return tensor


def plot_comparison(input_tensor, component_high, component_low, reconstructed,
                    output_directory="test_output"):
    """Plot visual comparison of input tensor, components, and reconstructed output."""
    odak.tools.check_directory(output_directory)

    # Input tensor
    odak.learn.tools.save_image(
        os.path.join(output_directory, "input_tensor.png"),
        input_tensor,
        cmin=0.0,
        cmax=1.0,
    )
    print("Saved test_output/input_tensor.png")

    # High component sum (sum across the last dimension)
    high_sum = component_high[..., 0] + component_high[..., 1]
    # Remove batch dimension if present
    if high_sum.dim() == 3:
        # Add a channel dimension if needed for save_image
        high_img = high_sum[0].unsqueeze(0)
    elif high_sum.dim() == 2:
        high_img = high_sum.unsqueeze(0)
    else:
        high_img = high_sum
    odak.learn.tools.save_image(
        os.path.join(output_directory, "component_high_sum.png"),
        high_img,
        cmin=0.0,
        cmax=2.0,
    )
    print("Saved test_output/component_high_sum.png")

    # Low component sum
    low_sum = component_low[..., 0] + component_low[..., 1]
    if low_sum.dim() == 3:
        low_img = low_sum[0].unsqueeze(0)
    elif low_sum.dim() == 2:
        low_img = low_sum.unsqueeze(0)
    else:
        low_img = low_sum
    odak.learn.tools.save_image(
        os.path.join(output_directory, "component_low_sum.png"),
        low_img,
        cmin=0.0,
        cmax=2.0,
    )
    print("Saved test_output/component_low_sum.png")

    # Reconstructed tensor
    odak.learn.tools.save_image(
        os.path.join(output_directory, "reconstructed_tensor.png"),
        reconstructed,
        cmin=0.0,
        cmax=1.0,
    )
    print("Saved test_output/reconstructed_tensor.png")


def test():
    device = torch.device("cpu")
    output_directory = "test_output"

    resolution = [16, 16]
    input_tensor = create_test_tensor(resolution).to(device)

    print("Input tensor shape:", input_tensor.shape)
    print("Input tensor:\n", input_tensor)

    component_high, component_low = decompose_double_phase(input_tensor)

    print("\nComponent high shape:", component_high.shape)
    print("Component high:\n", component_high)

    print("\nComponent low shape:", component_low.shape)
    print("Component low:\n", component_low)

    reconstructed = compose_double_phase(component_high, component_low)

    print("\nReconstructed tensor shape:", reconstructed.shape)
    print("Reconstructed tensor:\n", reconstructed)

    assert input_tensor.shape == reconstructed.shape, (
        f"Shape mismatch: {input_tensor.shape} vs {reconstructed.shape}"
    )
    assert torch.allclose(input_tensor, reconstructed), "Input and reconstructed tensors are not equal!"

    plot_comparison(input_tensor, component_high, component_low, reconstructed, output_directory)

    print("\nAll checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(test())
