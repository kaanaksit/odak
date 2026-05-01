import torch
import os
import sys
import odak
from odak.learn.tools import (
    decompose_wavelet_like, reconstruct_wavelet_like,
    save_image, load_image, save_torch_tensor
)
from odak.learn.perception import PSNR, SSIM
from odak.log import logger


def test():
    """
    Test wavelet-like decomposition for phase holograms.
    Achieves perfect reconstruction by storing detail residuals at each scale.
    Uses n_scales=3 which provides perfect reconstruction.
    """
    header = "test/test_learn_tools_bspline_decomposition.py"
    
    # Use CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("{} -> Using device: {}".format(header, device))
    
    output_dir = "test_output"
    odak.tools.check_directory(output_dir)
    
    logger.info("{} -> Wavelet-like Decomposition Test for Phase Holograms".format(header))
    
    # Load the phase hologram
    loading_failed = True
    try:
        phase_img = load_image('test/data/sample_direct_phase_hologram.png', normalizeby=255.0)
        if phase_img.dim() == 2 and phase_img.shape[0] == phase_img.shape[1]:
            logger.info("{} -> Loaded shape: {}".format(header, phase_img.shape))
            loading_failed = False
    except:
        pass
    
    if loading_failed:
        logger.info("{} -> Using synthetic test data".format(header))
        H, W = 512, 512
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        x_norm = 2 * (x - W/2) / max(H, W)
        y_norm = 2 * (y - H/2) / max(H, W)
        phase_img = (
            0.5 * torch.sin(3 * torch.pi * x_norm) +
            0.3 * torch.cos(3 * torch.pi * y_norm) +
            0.2 * torch.sin(6 * torch.pi * torch.sqrt(x_norm**2 + y_norm**2 + 1e-6))
        )
        phase_img = (phase_img - phase_img.min()) / (phase_img.max() - phase_img.min())
        logger.info("{} -> Created synthetic shape: {}".format(header, phase_img.shape))
    
    original_phase = phase_img.clone().to(device)
    if original_phase.dim() == 3:
        original_phase = original_phase.squeeze(0)
    H, W = original_phase.shape
    logger.info("{} -> Processed shape: {}".format(header, original_phase.shape))


    
    # Test wavelet-like decomposition with n_scales=3
    logger.info("{} -> Testing wavelet-like decomposition (n_scales=3)".format(header))
    
    n_scales = 3
    logger.info("{} -> Decomposing with {} scales...".format(header, n_scales))
    
    coefficients, residuals, base = decompose_wavelet_like(
        original_phase, n_scales=n_scales
    )
    
    reconstructed = reconstruct_wavelet_like(coefficients, residuals, base)
    
    # Compute metrics using odak perception tools
    psnr_metric = PSNR().to(device)
    ssim_metric = SSIM().to(device)
    
    # Add batch and channel dimensions for the metric functions
    original_exp = original_phase.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    recon_exp = reconstructed.unsqueeze(0).unsqueeze(0)
    
    psnr = psnr_metric(original_exp, recon_exp)
    ssim_like = ssim_metric(original_exp, recon_exp)
    
    # Compute MSE using torch's built-in loss function
    mse_loss_fn = torch.nn.MSELoss(reduction='mean')
    mse = mse_loss_fn(original_phase, reconstructed)
    
    total_data = sum(len(c) for c in coefficients) + sum(r.numel() for r in residuals) + base.numel()
    
    logger.info("{} -> Total data points: {}".format(header, total_data))
    logger.info("{} -> MSE: {:.10e}".format(header, mse))
    logger.info("{} -> PSNR: {:.2f} dB".format(header, psnr))
    logger.info("{} -> SSIM: {:.4f}".format(header, ssim_like))
    
    # Save results    
    logger.info("{} -> Saving decomposition results...".format(header))
    save_image(
        os.path.join(output_dir, "original_phase.png"),
        original_phase.unsqueeze(0),
        cmin=0.0,
        cmax=1.0,
    )

    save_image(
        os.path.join(output_dir, "wavelet_reconstruction.png"),
        reconstructed.unsqueeze(0),
        cmin=0.0,
        cmax=1.0,
    )
    
    diff = torch.abs(original_phase - reconstructed)
    save_image(
        os.path.join(output_dir, "wavelet_difference.png"),
        diff.unsqueeze(0),
        cmin=0, cmax=float(diff.max()) if diff.max() > 0 else 1e-10
    )
    
    # Save decomposition components as torch tensors
    save_torch_tensor(os.path.join(output_dir, "wavelet_coeffs.pt"), coefficients)
    save_torch_tensor(os.path.join(output_dir, "wavelet_residuals.pt"), residuals)
    save_torch_tensor(os.path.join(output_dir, "wavelet_base.pt"), base)
    
    # Visualize B-spline coefficients and residuals using diagrams
    logger.info("{} -> Generating visualization diagrams...".format(header))
    
    # Prepare all fields for visualization
    all_fields = []
    titles = []
    
    # Add base approximation
    all_fields.append(base)
    titles.append("Base Approximation")
    
    # Add residuals for each scale
    for scale_idx, residual in enumerate(residuals):
        all_fields.append(residual)
        titles.append("Residual Scale {}".format(scale_idx + 1))
    
    # Add coefficient grids (need to reshape to 2D for visualization)
    for scale_idx, coeffs in enumerate(coefficients):
        n_cx = int(coeffs.shape[0] ** 0.5)
        if n_cx * n_cx == coeffs.shape[0]:
            coef_grid = coeffs.view(n_cx, n_cx)
        else:
            # If not a perfect square, reshape to the nearest square or use view with -1
            coef_grid = coeffs.view(coeffs.shape[0], 1)
        
        all_fields.append(coef_grid)
        titles.append("Coeffs Scale {}".format(scale_idx + 1))
    
    # Create diagram
    fields_len = len(all_fields)
    diagram = odak.visualize.plotly.plot2dshow(
        title="Wavelet-like Decomposition (n_scales=3)",
        row_titles=titles,
        subplot_titles=[],
        rows=fields_len,
        cols=1,
        shape=[512, 600 * fields_len],
        color_scale='Inferno',
    )
    
    for field_id, field in enumerate(all_fields):
        diagram.add_field(
            field=field,
            row=field_id + 1,
            col=1,
            showscale=True,
        )
    
    # Show the diagram (for interactive environments)
    # diagram.show()
    
    # Also save individual components as images for easy viewing
    logger.info("{} -> Saving diagnostic images...".format(header))
    
    # Save base approximation
    save_image(
        os.path.join(output_dir, "wavelet_base.png"),
        base.unsqueeze(0),
        cmin=0.0,
        cmax=1.0,
    )
    
    # Save residuals
    for scale_idx, residual in enumerate(residuals):
        save_image(
            os.path.join(output_dir, "wavelet_residual_scale_{}.png".format(scale_idx + 1)),
            residual.unsqueeze(0),
            cmin=0.0,
            cmax=1.0,
        )
    
    # Save coefficient grids
    for scale_idx, coeffs in enumerate(coefficients):
        n_cx = int(coeffs.shape[0] ** 0.5)
        if n_cx * n_cx == coeffs.shape[0]:
            coef_grid = coeffs.view(n_cx, n_cx)
        else:
            coef_grid = coeffs.view(coeffs.shape[0], 1)
        
        # Normalize for visualization
        coef_norm = (coef_grid - coef_grid.min()) / (coef_grid.max() - coef_grid.min() + 1e-10)
        save_image(
            os.path.join(output_dir, "wavelet_coeffs_scale_{}.png".format(scale_idx + 1)),
            coef_norm.unsqueeze(0),
            cmin=0.0,
            cmax=1.0,
        )
    
    logger.info("{} -> Saved decomposition components and visualizations".format(header))
    
    # Generate summary
    logger.info("{} -> Summary:".format(header))
    summary_lines = [
        "=" * 60,
        "Wavelet-like Decomposition Summary",
        "=" * 60,
        "Method: decomposition_wavelet_like(n_scales={})".format(n_scales),
        "Image size: {} x {}".format(H, W),
        "Data stored: {} elements".format(total_data),
        "",
        "Reconstruction Quality:",
        "  MSE:  {:.10e}".format(mse),
        "  PSNR: {:.2f} dB".format(psnr),
        "  SSIM: {:.4f}".format(ssim_like),
        "",
        "The wavelet-like decomposition achieves perfect reconstruction",
        "by storing detail residuals at each decomposition scale.",
        "=" * 60,
    ]
    
    summary_text = "\n".join(summary_lines)
    logger.info(summary_text)
    
    with open(os.path.join(output_dir, "decomposition_summary.txt"), "w") as f:
        f.write(summary_text)
    
    assert mse < 1e-10, f"MSE {mse} should be near zero for perfect reconstruction"
    logger.info("{} -> All checks passed!".format(header))


if __name__ == "__main__":
    sys.exit(test())
