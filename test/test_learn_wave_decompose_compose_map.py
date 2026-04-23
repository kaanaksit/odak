import sys
import os
import torch
import numpy as np
import odak
from odak.learn.wave import (
    generate_decompose_index_map,
    decompose_map,
    compose_map,
    decompose_double_phase,
    compose_double_phase,
)


def plot_comparison(input_tensor, component_high, component_low, reconstructed,
                    decompose_map_indices, output_directory="test_output"):
    """Plot visual comparison of input tensor, components, and reconstructed output."""
    odak.tools.check_directory(output_directory)

    indices_high, indices_low = decompose_map_indices

    # Input tensor
    odak.learn.tools.save_image(
        os.path.join(output_directory, "input_tensor_map.png"),
        input_tensor,
        cmin=0.0,
        cmax=1.0,
    )
    print("Saved test_output/input_tensor_map.png")

    # Component high (2D image at half width)
    if component_high.dim() == 3:
        high_img = component_high[0].unsqueeze(0)
    else:
        high_img = component_high.unsqueeze(0)
    odak.learn.tools.save_image(
        os.path.join(output_directory, "component_high_map.png"),
        high_img,
        cmin=component_high.min().item(),
        cmax=component_high.max().item(),
    )
    print("Saved test_output/component_high_map.png")

    # Component low (2D image at half width)
    if component_low.dim() == 3:
        low_img = component_low[0].unsqueeze(0)
    else:
        low_img = component_low.unsqueeze(0)
    odak.learn.tools.save_image(
        os.path.join(output_directory, "component_low_map.png"),
        low_img,
        cmin=component_low.min().item(),
        cmax=component_low.max().item(),
    )
    print("Saved test_output/component_low_map.png")

    # Reconstructed tensor
    odak.learn.tools.save_image(
        os.path.join(output_directory, "reconstructed_tensor_map.png"),
        reconstructed,
        cmin=0.0,
        cmax=1.0,
    )
    print("Saved test_output/reconstructed_tensor_map.png")
    
    # Reshape index maps for visualization
    h_comp = input_tensor.shape[0]
    w_comp = input_tensor.shape[1] // 2
    indices_high_2d = indices_high.reshape(h_comp, w_comp).float()
    indices_low_2d = indices_low.reshape(h_comp, w_comp).float()
    
    # Save index maps with correct normalization
    max_high = indices_high.max().item()
    max_low = indices_low.max().item()
    
    odak.learn.tools.save_image(
        os.path.join(output_directory, "decompose_map_high_indices.png"),
        indices_high_2d.unsqueeze(0),
        cmin=0.0,
        cmax=max_high,
    )
    print("Saved test_output/decompose_map_high_indices.png (cmax={})".format(max_high))
    
    odak.learn.tools.save_image(
        os.path.join(output_directory, "decompose_map_low_indices.png"),
        indices_low_2d.unsqueeze(0),
        cmin=0.0,
        cmax=max_low,
    )
    print("Saved test_output/decompose_map_low_indices.png (cmax={})".format(max_low))
    
    # Print sample of index values to verify correctness
    print("\nSample from decompose map (high indices):")
    print("  Row 0 (even), cols 0-3:", indices_high_2d[0, :4].tolist())
    print("  Row 1 (odd), cols 0-3:", indices_high_2d[1, :4].tolist())
    print("  Row 2 (even), cols 0-3:", indices_high_2d[2, :4].tolist())
    print("  Row 3 (odd), cols 0-3:", indices_high_2d[3, :4].tolist())
    
    print("\nSample from decompose map (low indices):")
    print("  Row 0 (even), cols 0-3:", indices_low_2d[0, :4].tolist())
    print("  Row 1 (odd), cols 0-3:", indices_low_2d[1, :4].tolist())
    print("  Row 2 (even), cols 0-3:", indices_low_2d[2, :4].tolist())
    print("  Row 3 (odd), cols 0-3:", indices_low_2d[3, :4].tolist())


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

    # Generate index maps for visualization only
    h, w = input_tensor.shape
    decompose_map_indices = generate_decompose_index_map(h, w, device)

    # Decompose - now returns compose_map directly!
    component_high, component_low, compose_map_from_decomp = decompose_map(input_tensor)

    print("\nComponent high shape:", component_high.shape)
    print("Component low shape:", component_low.shape)
    print("Compose map from decompose_map:", compose_map_from_decomp[0].shape, compose_map_from_decomp[1].shape)

    # Compose using the compose_map returned from decompose_map
    reconstructed = compose_map(component_high, component_low, compose_map_from_decomp)

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

    plot_comparison(input_tensor, component_high, component_low, reconstructed,
                   decompose_map_indices, output_directory)

    # Test with batch dimension
    print("\n--- Testing with batch dimension ---")
    batch_input = input_tensor.unsqueeze(0).expand(3, -1, -1)
    print("Batch input shape:", batch_input.shape)

    batch_comp_high, batch_comp_low, batch_compose_map = decompose_map(batch_input)
    print("Batch component high shape:", batch_comp_high.shape)
    print("Batch component low shape:", batch_comp_low.shape)

    batch_reconstructed = compose_map(batch_comp_high, batch_comp_low, batch_compose_map)
    print("Batch reconstructed shape:", batch_reconstructed.shape)

    assert batch_input.shape == batch_reconstructed.shape
    assert torch.allclose(batch_input, batch_reconstructed)

    # For visualization, regenerate the index map (same for all batches)
    batch_viz_decomp_map = generate_decompose_index_map(h, w, device)
    plot_comparison(batch_input[0], batch_comp_high[0], batch_comp_low[0], 
                   batch_reconstructed[0], batch_viz_decomp_map, output_directory)

    print("\nAll checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(test())
