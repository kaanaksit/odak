import sys
import os
import torch
import numpy as np
import odak
from odak.learn.wave import (
    generate_decompose_index_map,
    generate_compose_index_map,
    decompose_map,
    compose_map,
    decompose_double_phase,
    compose_double_phase,
)


def plot_comparison(input_tensor, component_high, component_low, reconstructed,
                    output_directory="test_output"):
    """Plot visual comparison of input tensor, components, and reconstructed output."""
    odak.tools.check_directory(output_directory)

    # Input tensor
    odak.learn.tools.save_image(
        os.path.join(output_directory, "input_tensor_map.png"),
        input_tensor,
        cmin=0.0,
        cmax=1.0,
    )
    print("Saved test_output/input_tensor_map.png")

    # High component sum
    high_sum = component_high[..., 0] + component_high[..., 1]
    if high_sum.dim() == 3:
        high_img = high_sum[0].unsqueeze(0)
    elif high_sum.dim() == 2:
        high_img = high_sum.unsqueeze(0)
    else:
        high_img = high_sum
    odak.learn.tools.save_image(
        os.path.join(output_directory, "component_high_sum_map.png"),
        high_img,
        cmin=0.0,
        cmax=2.0,
    )
    print("Saved test_output/component_high_sum_map.png")

    # Low component sum
    low_sum = component_low[..., 0] + component_low[..., 1]
    if low_sum.dim() == 3:
        low_img = low_sum[0].unsqueeze(0)
    elif low_sum.dim() == 2:
        low_img = low_sum.unsqueeze(0)
    else:
        low_img = low_sum
    odak.learn.tools.save_image(
        os.path.join(output_directory, "component_low_sum_map.png"),
        low_img,
        cmin=0.0,
        cmax=2.0,
    )
    print("Saved test_output/component_low_sum_map.png")

    # Reconstructed tensor
    odak.learn.tools.save_image(
        os.path.join(output_directory, "reconstructed_tensor_map.png"),
        reconstructed,
        cmin=0.0,
        cmax=1.0,
    )
    print("Saved test_output/reconstructed_tensor_map.png")


def test():
    device = torch.device("cpu")
    output_directory = "test_output"

    resolution = [16, 16]
    input_tensor = odak.learn.tools.load_image(
        'test/data/sample_hologram.png',
        normalizeby=255.0,
        torch_style=True
    )[0].to(device)

    print("Input tensor shape:", input_tensor.shape)

    # Generate index maps
    h, w = input_tensor.shape
    decompose_map_indices = generate_decompose_index_map(h, w, device)
    compose_map_indices = generate_compose_index_map(h, w, device)

    print("Decompose map indices_high shape:", decompose_map_indices[0].shape)
    print("Decompose map indices_low shape:", decompose_map_indices[1].shape)
    print("Compose map indices_high shape:", compose_map_indices[0].shape)
    print("Compose map indices_low shape:", compose_map_indices[1].shape)

    # Decompose using maps
    component_high, component_low = decompose_map(input_tensor, decompose_map_indices)

    print("\nComponent high shape:", component_high.shape)
    print("Component low shape:", component_low.shape)

    # Compose using maps
    reconstructed = compose_map(component_high, component_low, compose_map_indices)

    print("\nReconstructed shape:", reconstructed.shape)

    # Verify reconstruction
    assert input_tensor.shape == reconstructed.shape, (
        f"Shape mismatch: {input_tensor.shape} vs {reconstructed.shape}"
    )
    assert torch.allclose(input_tensor, reconstructed), \
        "Input and reconstructed tensors are not equal!"

    # Compare with decompose_double_phase
    comp_high_orig, comp_low_orig = decompose_double_phase(input_tensor)
    assert torch.allclose(component_high, comp_high_orig), \
        "decompose_map high differs from decompose_double_phase!"
    assert torch.allclose(component_low, comp_low_orig), \
        "decompose_map low differs from decompose_double_phase!"

    reconstructed_orig = compose_double_phase(comp_high_orig, comp_low_orig)
    assert torch.allclose(reconstructed, reconstructed_orig), \
        "compose_map differs from compose_double_phase!"

    plot_comparison(input_tensor, component_high, component_low, reconstructed, output_directory)

    # Test with batch dimension
    print("\n--- Testing with batch dimension ---")
    batch_input = input_tensor.unsqueeze(0).expand(3, -1, -1)
    print("Batch input shape:", batch_input.shape)

    batch_comp_high, batch_comp_low = decompose_map(batch_input, decompose_map_indices)
    print("Batch component high shape:", batch_comp_high.shape)
    print("Batch component low shape:", batch_comp_low.shape)

    batch_reconstructed = compose_map(batch_comp_high, batch_comp_low, compose_map_indices)
    print("Batch reconstructed shape:", batch_reconstructed.shape)

    assert batch_input.shape == batch_reconstructed.shape
    assert torch.allclose(batch_input, batch_reconstructed)

    plot_comparison(batch_input[0], batch_comp_high[0], batch_comp_low[0], 
                   batch_reconstructed[0], output_directory)

    print("\nAll checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(test())
