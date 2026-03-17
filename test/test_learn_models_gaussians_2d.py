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
                         Number of training epochs (default: 50).
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
    
    # Load test image using odak.learn.tools.load_image (returns torch tensor)
    image = odak.learn.tools.load_image(
        "test/data/fruit_lady.png", 
        normalizeby=255.0,  # Normalize to [0, 1]
        torch_style=False   # Keep as (H, W, C)
    )

    # Get image dimensions directly from tensor shape
    if len(image.shape) == 3:
        height, width = image.shape[0], image.shape[1]
        # Convert RGB to grayscale using standard formula (all in torch)
        ground_truth = (
            0.299 * image[:, :, 0] + 
            0.587 * image[:, :, 1] + 
            0.114 * image[:, :, 2]
        )
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

    logger.info("Number of training pixels: {}".format(X.shape[0] * X.shape[1]))

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

        # Forward pass (gaussians_2d already sums and adds dimension)
        estimates = model(X, Y)  # Shape: (height, width, 1)
        
        # Remove last dimension to match ground_truth shape (HxW)
        estimates_squeezed = estimates.squeeze(-1)

        # Compute loss
        loss_l2 = l2_loss(estimates_squeezed, ground_truth)
        loss_l1 = l1_loss(estimates_squeezed, ground_truth)
        loss = 1.0 * loss_l2 + 0.1 * loss_l1

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss
        loss_value = loss.item()
        loss_history.append(loss_value)

        # Update progress bar description
        description = "gaussian_2d overfit - L2:{:.6f}".format(loss_l2.item())
        t_epoch.set_description(description)

    logger.info("Final loss: {:.6f}".format(loss_value))

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

    # Generate final reconstruction for visualization
    with torch.no_grad():
        final_reconstruction = model(X, Y)
        final_reconstruction = final_reconstruction.squeeze(-1)  # Remove last dim

    if visualize:
        import plotly.graph_objects as go

        # Prepare data for visualization (keep in torch, convert only at viz time)
        img_original = ground_truth.cpu()  # Already HxW tensor
        reconstruction = final_reconstruction.cpu().clamp(0, 1)  # Clip to [0, 1]

        fig = go.Figure()

        # Add original image using Heatmap (Plotly accepts numpy arrays)
        fig.add_trace(
            go.Heatmap(
                z=img_original.T.numpy(),
                x=list(range(img_original.shape[1])),
                y=list(range(img_original.shape[0])),
                name="Original",
                showscale=True,
                colorbar=dict(title="Original"),
                colorscale="Greys",
                opacity=1.0,
            )
        )

        # Add reconstructed image  
        fig.add_trace(
            go.Heatmap(
                z=reconstruction.T.numpy(),
                x=list(range(reconstruction.shape[1])),
                y=list(range(reconstruction.shape[0])),
                name="Reconstructed ({} Gaussians)".format(number_of_elements),
                showscale=True,
                colorbar=dict(title="Reconstructed"),
                colorscale="Viridis",
                opacity=0.6,
            )
        )

        fig.update_layout(
            title="2D Gaussian Model Overfit Test (Full Resolution)",
            height=500,
            width=1000,
        )

        # Show the figure
        # Note: To disable visualization, run with visualize=False
        if "pytest" in sys.modules or "pytest" in str(sys.argv):
            # In pytest mode, skip interactive display
            logger.info("Visualization skipped during pytest execution.")
        else:
            fig.show()

    logger.info("Test completed successfully.")
    assert True


if __name__ == "__main__":
    import odak

    # Initialize test output directory  
    odak.tools.check_directory("test_output")

    # Run the test
    test_gaussian_2d_overfit(
        number_of_elements=15,
        learning_rate=0.05,
        number_of_epochs=100,
        visualize=False,  # Disable visualization when running as script to simplify logs
    )
