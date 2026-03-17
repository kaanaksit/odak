"""
Unit tests for the gaussian_2d model.

This module contains tests for the 2D Gaussian model that overfits to
sampled pixel locations from an image.
All operations are performed in PyTorch tensors.
"""

import sys
import torch


def test_gaussian_2d_overfit(
    number_of_elements=10,
    learning_rate=1e-2,
    number_of_epochs=50,
    visualize=True,
    device=None,
):
    """
    Test the gaussian_2d model by overfitting to sampled pixel locations from an image.

    This test:
    1. Loads the fruit_lady.png image
    2. Creates coordinate grids normalized to [-1, 1]
    3. Extracts grayscale values from the image as ground truth
    4. Initializes a gaussian_2d model with given number of elements
    5. Optimizes the model to overfit the sampled pixel values
    6. Validates that loss decreases during training

    All operations are performed in PyTorch tensors without numpy conversion.

    Parameters
    ----------
    number_of_elements : int, optional
                        Number of 2D Gaussian primitives in the model (default: 10).
    learning_rate      : float, optional
                         Learning rate for AdamW optimizer (default: 1e-2).
    number_of_epochs   : int, optional
    visualize          : bool, optional
                         Whether to visualize the optimization process (default: True).
    device             : torch.device or None, optional
                         Device to run computation on. If None, uses CPU (default: None).

    Notes
    ----- 
    - Loss should decrease over time as the model learns to fit the data.
    - Visualization shows original image vs model reconstruction.
    """
    import odak
    from tqdm import tqdm
    from odak.log import logger

    if device is None:
        _device = torch.device("cpu")
    else:
        _device = device
    
    # Load test image using odak.learn.tools.load_image (returns torch tensor)
    image = odak.learn.tools.load_image(
        "test/data/fruit_lady.png", 
        normalizeby=255.0,  # Normalize to [0, 1]
        torch_style=False   # Keep as (H, W, C)
    )

    # Get image dimensions directly from tensor shape
    if len(image.shape) == 3:
        height, width = image.shape[0], image.shape[1]
        # Convert RGB to grayscale using odak utility
        ground_truth = odak.learn.perception.rgb_to_gray(image)
    else:
        height, width = image.shape[0], image.shape[1]
        ground_truth = image.clone()

    # Move to device (ground_truth is already HxW, no extra dimension needed for 2D model)
    ground_truth = ground_truth.to(_device)

    logger.info("Image loaded: {}x{} pixels (no resizing)".format(width, height))

    # Create coordinate grids normalized to [-1, 1] (no resizing)
    x = torch.linspace(1.0, -1.0, width, device=_device)
    y = torch.linspace(-1.0, 1.0, height, device=_device)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Reshape X and Y to (-1, 1) for batch processing
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)

    logger.info("Number of training pixels: {}".format(X_flat.shape[0]))

    # Initialize the wrapper model (not the primitive)
    model = odak.learn.models.gaussians_2d(number_of_elements=number_of_elements)
    model = model.to(_device)

    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: {}".format(total_parameters))

    # Loss functions
    l2_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=number_of_epochs,
        power=1.0,
        last_epoch=-1,
    )

    # Training loop
    loss_history = []
    t_epoch = tqdm(range(number_of_epochs), leave=False, dynamic_ncols=True)

    for epoch_id in t_epoch:
        optimizer.zero_grad()

        # Forward pass: primitive receives (N, 1), returns (N, number_of_elements)
        # but wrapper sums over elements and adds dimension → (N, 1)
        estimates = model(X_flat, Y_flat)  # Shape: (N, 1) from wrapper
        
        # Remove the last dimension and reshape back to original image dimensions
        estimates_squeezed = estimates.squeeze(-1)  # Shape: (N,)
        estimates_reshaped = estimates_squeezed.reshape(height, width)  # Shape: (H, W)

        # Compute loss (both are HxW tensors now)
        loss_l2 = l2_loss(estimates_reshaped, ground_truth)
        loss_l1 = l1_loss(estimates_reshaped, ground_truth)
        loss = 1.0 * loss_l2 + 0.1 * loss_l1

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss
        loss_value = loss.item()
        loss_history.append(loss_value)

        # Update progress bar description with current L2 loss
        description = "gaussian_2d overfit - L2:{:.6f}".format(loss_l2.item())
        t_epoch.set_description(description)

    # Assert that loss decreased (training made progress)
    if len(loss_history) > 1:
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        assert final_loss < initial_loss, (
            "Model failed to learn: final loss ({:.6f}) >= initial loss ({:.6f})".format(
                final_loss, initial_loss
            )
        )
        logger.info(
            "Loss reduced from {:.6f} to {:.6f} ({:.1f}% improvement)".format(
                initial_loss,
                final_loss,
                (initial_loss - final_loss) / initial_loss * 100.0,
            )
        )

    # Generate final reconstruction for visualization (use flattened coords like training)
    with torch.no_grad():
        final_reconstruction = model(X_flat, Y_flat)  # Shape: (N, 1) from wrapper
        final_reconstruction = final_reconstruction.squeeze(-1).reshape(height, width)  # Back to (H, W)
        
        # Clip to valid range for display
        final_reconstruction = final_reconstruction.clamp(0, 1)

    if visualize:
        # Prepare visualization using odak.visualize.plotly.plot2dshow
        fields = [ground_truth, final_reconstruction]
        row_titles = ["Original", "Reconstructed ({} Gaussians)".format(number_of_elements)]
        
        diagram = odak.visualize.plotly.plot2dshow(
            title="2D Gaussian Model Overfit Test",
            row_titles=row_titles,
            subplot_titles=[],
            rows=2,
            cols=1,
            shape=[512, 600 * 2],
        )
        
        for field_id, field in enumerate(fields):
            diagram.add_field(
                field=torch.flipud(field),
                row=field_id + 1,
                col=1,
                showscale=False,
            )
        
        # Show only if not running under pytest
        if "pytest" not in sys.modules and "pytest" not in str(sys.argv):
            diagram.show()
        else:
            logger.info("Visualization skipped during pytest execution.")

    logger.info("Test completed successfully.")
    assert True


if __name__ == "__main__":
    import odak

    # Initialize test output directory  
    odak.tools.check_directory("test_output")

    # Run the test
    test_gaussian_2d_overfit(
        number_of_elements=2700,
        learning_rate=5e-3,
        number_of_epochs=10, # Set it to 10000 for a full optimization
        visualize=True,
        device=torch.device('cuda'),
    )
