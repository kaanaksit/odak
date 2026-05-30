import torch
import torch.nn


def quantize(image_field, bits=8, limits=[0.0, 1.0]):
    """
    Quantize an image field to a specified number of bits.
    
    This function maps the input image field from its original range to a quantized 
    representation with the specified number of bits.

    Parameters
    ----------
    image_field : torch.tensor
                  Input image field between any range.
    bits        : int
                  Number of bits for quantization (1-8).
    limits      : list
                  The minimum and maximum of the image_field variable.

    Returns
    ----------
    quantized_field   : torch.tensor
                        Quantized image field.
    """
    normalized_field = (image_field - limits[0]) / (limits[1] - limits[0])
    divider = 2**bits
    quantized_field = normalized_field * divider
    quantized_field = quantized_field.int()
    return quantized_field


def smooth_pad(field, size=None, method="center", smooth_factor=[0.5, 0.5]):
    """
    Smooth pad a field to double its size or specified size with gradual transition.
    
    This function pads a field with a smooth transition from the content edge to zero,
    avoiding sharp edges. The transition rate is controllable along X and Y axes.
    The original content area is preserved at full value, and the padding region shows
    a smooth falloff from the content edge to zero at the canvas boundary.
    
    Parameters
    ----------
    field             : torch.tensor
                        Input field MxN, CxMxN, 1xMxN, or 1x1xMxN array.
    size              : list
                        Size to be smoothpadded (e.g., [m, n], last two dimensions only). 
                        If None, doubles the last two dimensions.
    method            : str
                        Smoothpad either by placing the content to center or to the left.
    smooth_factor     : list
                        Transition smoothness factor along X and Y axes [fx, fy].
                        Values in range [0, 1] control the falloff rate:
                          0 = gentle (slow falloff, high value at midpoint of padding)
                          1 = steep (fast falloff, low value at midpoint of padding)
                        Example: [0.2, 0.6] gives gentle X-axis falloff and steeper Y-axis falloff.

    Returns
    ----------
    field_smooth_padded : torch.tensor
                          Smoothpadded version of the input field with gradual falloff.
    """
    orig_resolution = field.shape
    orig_ndim = len(orig_resolution)
    
    channels_first = True
    if orig_ndim == 3 and orig_resolution[-1] == 3:
        channels_first = False
    elif orig_ndim == 4 and orig_resolution[-1] == 3:
        channels_first = False
    
    if orig_ndim == 2:
        field = field.unsqueeze(0)
    elif orig_ndim == 3:
        if orig_resolution[0] == 1:
            field = field.squeeze(0)
            field = field.unsqueeze(0)
        elif not channels_first:
            field = field.permute(2, 0, 1)
    elif orig_ndim == 4:
        if orig_resolution[0] == 1:
            field = field.squeeze(0)
            if not channels_first:
                field = field.permute(2, 0, 1)
        elif not channels_first:
            field = field.permute(0, 3, 1, 2)
    
    if len(field.shape) < 4:
        field = field.unsqueeze(0)
    
    if size is None:
        resolution = [
            field.shape[0],
            field.shape[1],
            2 * field.shape[-2],
            2 * field.shape[-1],
        ]
    else:
        resolution = [field.shape[0], field.shape[1], size[0], size[1]]
    
    h, w = resolution[-2], resolution[-1]
    field_h, field_w = field.shape[-2], field.shape[-1]
    sx, sy = smooth_factor
    
    # Place content in the center
    field_smooth_padded = torch.zeros(resolution, device=field.device, dtype=field.dtype)
    
    if method == "center":
        start_y = h // 2 - field_h // 2
        start_x = w // 2 - field_w // 2
    elif method == "left":
        start_y, start_x = 0, 0
    
    field_smooth_padded[
        :, :,
        start_y : start_y + field_h,
        start_x : start_x + field_w,
    ] = field
    
    # For smooth falloff, we need to extend content values into padding region
    # Create 1D falloff functions for each axis
    def create_falloff_1d(size, content_start, content_end, factor):
        """Create 1D falloff array that is 1 in content region and fades to 0 at edges."""
        result = torch.ones(size, device=field.device, dtype=field.dtype)
        
        # Left margin
        left_margin = content_start
        if left_margin > 0:
            for i in range(left_margin):
                # Distance from content edge (content_start - 1)
                dist = content_start - i
                # Normalize: at dist=1 (adjacent to content), t=1 giving value=1; at dist=left_margin (edge), t=0 giving value=0
                t = (dist - 1) / (left_margin - 1) if left_margin > 1 else 1.0
                result[i] = torch.cos(torch.pi * (1 - t) * factor / 2.0) ** 2
        
        # Right margin
        right_margin = size - content_end
        if right_margin > 0:
            for i in range(content_end, size):
                dist = i - content_end + 1
                t = (dist - 1) / (right_margin - 1) if right_margin > 1 else 1.0
                result[i] = torch.cos(torch.pi * (1 - t) * factor / 2.0) ** 2
        
        return result
    
    # Actually, let me think about this more carefully
    # The falloff should be: 1 at content edge, 0 at canvas edge
    # For left margin: pixel at position i (0 <= i < content_start)
    #   Distance from content edge = content_start - i
    #   We want: at i = content_start - 1 (adjacent), value = 1
    #            at i = 0 (edge), value = 0
    #   So: t = (content_start - i - 1) / (content_start - 1)
    #       value = cos(pi * t / 2)^2
    #       At i = content_start - 1: t = 0, value = 1
    #       At i = 0: t = 1, value = 0
    
    y_falloff = torch.ones(h, device=field.device, dtype=field.dtype)
    x_falloff = torch.ones(w, device=field.device, dtype=field.dtype)
    
    def get_falloff_value(t, factor):
        """Get falloff value: 1 at t=0 (content edge), 0 at t=1 (canvas edge).
        
        smooth_factor in [0, 1] controls falloff steepness using exponential mapping:
        effective_exp = 100^factor
        
        Midpoint (t=0.5) falloff values:
          factor=0.0 -> 1.0000 (no falloff)
          factor=0.1 -> 0.58   (very gentle)
          factor=0.2 -> 0.42   (gentle)
          factor=0.3 -> 0.25   (moderate-gentle)
          factor=0.4 -> 0.11   (moderate)
          factor=0.5 -> 0.03   (moderate-steep)
          factor=0.6 -> 0.004  (steep)
          factor>=0.7 -> ~0    (very steep)
        """
        base = torch.cos(torch.pi * t / 2.0).clamp(min=0)
        # Exponential mapping: 100^factor gives good visual spread
        if factor <= 0:
            effective_exp = 0.0
        else:
            effective_exp = 100.0 ** factor
        result = base ** effective_exp
        # Debug print for t=0.5025 case
        if abs(t.item() - 0.5025) < 0.001:
            print(f"  get_falloff_value: t={t.item():.4f}, factor={factor:.4f}, base={base.item():.4f}, exp={effective_exp:.4f}, result={result.item():.4f}")
        return result
    
    # Left y margin
    if start_y > 0:
        for i in range(start_y):
            t = torch.tensor((start_y - i - 1) / (start_y - 1) if start_y > 1 else 1.0)
            y_falloff[i] = get_falloff_value(t, sy)
    
    # Right y margin
    if h - (start_y + field_h) > 0:
        for i in range(start_y + field_h, h):
            t = torch.tensor((i - (start_y + field_h)) / (h - start_y - field_h - 1) if (h - start_y - field_h) > 1 else 1.0)
            y_falloff[i] = get_falloff_value(t, sy)
    
    # Left x margin
    if start_x > 0:
        for i in range(start_x):
            t = torch.tensor((start_x - i - 1) / (start_x - 1) if start_x > 1 else 1.0)
            x_falloff[i] = get_falloff_value(t, sx)
    
    # Right x margin  
    if w - (start_x + field_w) > 0:
        for i in range(start_x + field_w, w):
            t = torch.tensor((i - (start_x + field_w)) / (w - start_x - field_w - 1) if (w - start_x - field_w) > 1 else 1.0)
            x_falloff[i] = get_falloff_value(t, sx)
    
    # Result tensor
    result = torch.zeros(resolution, device=field.device, dtype=field.dtype)
    
    # Copy content region
    result[:, :, start_y:start_y+field_h, start_x:start_x+field_w] = field
    
    # Extend top row upward (horizontal strip)
    if start_y > 0:
        for i in range(start_y - 1, -1, -1):
            t = torch.tensor((start_y - i - 1) / (start_y - 1) if start_y > 1 else 1.0)
            falloff = get_falloff_value(t, sy)
            result[:, :, i, start_x:start_x+field_w] = falloff * field[:, :, 0, :]
    
    # Extend bottom row downward
    if h - (start_y + field_h) > 0:
        for i in range(start_y + field_h, h):
            t_val = (i - (start_y + field_h)) / (h - start_y - field_h - 1) if (h - start_y - field_h) > 1 else 1.0
            t = torch.tensor(t_val)
            falloff = get_falloff_value(t, sy)
            if i == 500:  # Debug print for row 500
                print(f"DEBUG: i={i}, t={t_val:.4f}, falloff={falloff.item():.4f}, field[:, -1, 100]={field[0, 0, -1, 100].item():.4f}")
                print(f"DEBUG: Before assignment, result[0, 0, {i}, {start_x+100}] = {result[0, 0, i, start_x+100].item():.6f}")
            result[:, :, i, start_x:start_x+field_w] = falloff * field[:, :, -1, :]
            if i == 500:
                print(f"DEBUG: After bottom extension, result[0, 0, {i}, {start_x+100}] = {result[0, 0, i, start_x+100].item():.6f}")
    
    # Extend left column leftward (vertical strip)
    if start_x > 0:
        for i in range(start_x - 1, -1, -1):
            t = torch.tensor((start_x - i - 1) / (start_x - 1) if start_x > 1 else 1.0)
            falloff = get_falloff_value(t, sx)
            result[:, :, start_y:start_y+field_h, i] = falloff * field[:, :, :, 0]
    
    # Extend right column rightward
    if w - (start_x + field_w) > 0:
        for i in range(start_x + field_w, w):
            t = torch.tensor((i - (start_x + field_w)) / (w - start_x - field_w - 1) if (w - start_x - field_w) > 1 else 1.0)
            falloff = get_falloff_value(t, sx)
            result[:, :, start_y:start_y+field_h, i] = falloff * field[:, :, :, -1]
    
    # Fill corner regions with product of x and y falloffs
    # Top-left corner
    if start_y > 0 and start_x > 0:
        for yi in range(start_y):
            t_y = torch.tensor((start_y - yi - 1) / (start_y - 1) if start_y > 1 else 1.0)
            for xi in range(start_x):
                t_x = torch.tensor((start_x - xi - 1) / (start_x - 1) if start_x > 1 else 1.0)
                falloff = get_falloff_value(t_y, sy) * get_falloff_value(t_x, sx)
                result[:, :, yi, xi] = falloff * field[:, :, 0, 0]
    
    # Top-right corner
    if start_y > 0 and w - (start_x + field_w) > 0:
        for yi in range(start_y):
            t_y = torch.tensor((start_y - yi - 1) / (start_y - 1) if start_y > 1 else 1.0)
            for xi in range(start_x + field_w, w):
                t_x = torch.tensor((xi - (start_x + field_w)) / (w - start_x - field_w - 1) if (w - start_x - field_w) > 1 else 1.0)
                falloff = get_falloff_value(t_y, sy) * get_falloff_value(t_x, sx)
                result[:, :, yi, xi] = falloff * field[:, :, 0, -1]
    
    # Bottom-left corner
    if h - (start_y + field_h) > 0 and start_x > 0:
        for yi in range(start_y + field_h, h):
            t_y = torch.tensor((yi - (start_y + field_h)) / (h - start_y - field_h - 1) if (h - start_y - field_h) > 1 else 1.0)
            for xi in range(start_x):
                t_x = torch.tensor((start_x - xi - 1) / (start_x - 1) if start_x > 1 else 1.0)
                falloff = get_falloff_value(t_y, sy) * get_falloff_value(t_x, sx)
                result[:, :, yi, xi] = falloff * field[:, :, -1, 0]
    
    # Bottom-right corner
    if h - (start_y + field_h) > 0 and w - (start_x + field_w) > 0:
        for yi in range(start_y + field_h, h):
            t_y = torch.tensor((yi - (start_y + field_h)) / (h - start_y - field_h - 1) if (h - start_y - field_h) > 1 else 1.0)
            for xi in range(start_x + field_w, w):
                t_x = torch.tensor((xi - (start_x + field_w)) / (w - start_x - field_w - 1) if (w - start_x - field_w) > 1 else 1.0)
                falloff = get_falloff_value(t_y, sy) * get_falloff_value(t_x, sx)
                result[:, :, yi, xi] = falloff * field[:, :, -1, -1]
    
    if not channels_first:
        if len(orig_resolution) == 3:
            result = result.squeeze(0)
            result = result.permute(1, 2, 0)
        elif len(orig_resolution) == 4:
            result = result.permute(0, 2, 3, 1)
    
    if len(orig_resolution) == 2:
        result = result.squeeze(0).squeeze(0)
    elif len(orig_resolution) == 3 and channels_first:
        if orig_resolution[0] == 3:
            result = result.squeeze(0)
        elif orig_resolution[0] == 1:
            result = result.squeeze(0)
        else:
            result = result.squeeze(0)
    elif len(orig_resolution) == 4 and channels_first:
        if orig_resolution[0] == 1:
            result = result.squeeze(0)
    return result


def zero_pad(field, size=None, method="center"):
    """
    Zero pad a field to double its size or specified size.
    
    This function pads a field with zeros to either double its size (default) 
    or to a specified size. The input can be 2D, 3D or 4D tensors.
    
    Parameters
    ----------
    field             : torch.tensor
                        Input field MxN, CxMxN, 1xMxN, or 1x1xMxN array.
    size              : list
                        Size to be zeropadded (e.g., [m, n], last two dimensions only). 
                        If None, doubles the last two dimensions.
    method            : str
                        Zeropad either by placing the content to center or to the left.

    Returns
    ----------
    field_zero_padded : torch.tensor
                        Zeropadded version of the input field.
    """
    orig_resolution = field.shape
    orig_ndim = len(orig_resolution)
    
    channels_first = True
    if orig_ndim == 3 and orig_resolution[-1] == 3:
        channels_first = False
    elif orig_ndim == 4 and orig_resolution[-1] == 3:
        channels_first = False
    
    if orig_ndim == 2:
        field = field.unsqueeze(0)
    elif orig_ndim == 3:
        if orig_resolution[0] == 1:
            field = field.squeeze(0)
            field = field.unsqueeze(0)
        elif not channels_first:
            field = field.permute(2, 0, 1)
    elif orig_ndim == 4:
        if orig_resolution[0] == 1:
            field = field.squeeze(0)
            if not channels_first:
                field = field.permute(2, 0, 1)
        elif not channels_first:
            field = field.permute(0, 3, 1, 2)
    
    if len(field.shape) < 4:
        field = field.unsqueeze(0)
    
    if size is None:
        resolution = [
            field.shape[0],
            field.shape[1],
            2 * field.shape[-2],
            2 * field.shape[-1],
        ]
    else:
        resolution = [field.shape[0], field.shape[1], size[0], size[1]]
    field_zero_padded = torch.zeros(resolution, device=field.device, dtype=field.dtype)
    if method == "center":
        start = [
            resolution[-2] // 2 - field.shape[-2] // 2,
            resolution[-1] // 2 - field.shape[-1] // 2,
        ]
        field_zero_padded[
            :,
            :,
            start[0] : start[0] + field.shape[-2],
            start[1] : start[1] + field.shape[-1],
        ] = field
    elif method == "left":
        field_zero_padded[:, :, 0 : field.shape[-2], 0 : field.shape[-1]] = field
    
    if not channels_first:
        if len(orig_resolution) == 3:
            field_zero_padded = field_zero_padded.squeeze(0)
            field_zero_padded = field_zero_padded.permute(1, 2, 0)
        elif len(orig_resolution) == 4:
            field_zero_padded = field_zero_padded.permute(0, 2, 3, 1)
    
    if len(orig_resolution) == 2:
        field_zero_padded = field_zero_padded.squeeze(0).squeeze(0)
    elif len(orig_resolution) == 3 and channels_first:
        if orig_resolution[0] == 3:
            field_zero_padded = field_zero_padded.squeeze(0)
        elif orig_resolution[0] == 1:
            field_zero_padded = field_zero_padded.squeeze(0)
        else:
            field_zero_padded = field_zero_padded.squeeze(0)
    elif len(orig_resolution) == 4 and channels_first:
        if orig_resolution[0] == 1:
            field_zero_padded = field_zero_padded.squeeze(0)
    return field_zero_padded


def crop_center(field, size=None):
    """
    Crop the center of a field to specified size or half of current size.
    
    This function crops the center of a field to either half of its current size (default) 
    or to a specified size. The input can be 2D, 3D or 4D tensors.
    
    Parameters
    ----------
    field       : torch.tensor
                  Input field MxN, CxMxN, 1xMxN, or 1x1xMxN array.
    size        : list
                  Dimensions to crop with respect to center of the image (e.g., M x N or 1 x 1 x M x N).
                  If None, crops to half of the current size.

    Returns
    ----------
    cropped     : torch.tensor
                  Cropped version of the input field.
    """
    orig_resolution = field.shape
    orig_ndim = len(orig_resolution)
    
    channels_first = True
    if orig_ndim == 3 and orig_resolution[-1] == 3:
        channels_first = False
    elif orig_ndim == 4 and orig_resolution[-1] == 3:
        channels_first = False
    
    if orig_ndim == 2:
        field = field.unsqueeze(0)
    elif orig_ndim == 3:
        if orig_resolution[0] == 1:
            field = field.squeeze(0)
            field = field.unsqueeze(0)
        elif not channels_first:
            field = field.permute(2, 0, 1)
    elif orig_ndim == 4:
        if orig_resolution[0] == 1:
            field = field.squeeze(0)
            if not channels_first:
                field = field.permute(2, 0, 1)
        elif not channels_first:
            field = field.permute(0, 3, 1, 2)
    
    if len(field.shape) < 4:
        field = field.unsqueeze(0)
    
    if size is None:
        qx = int(field.shape[-2] // 4)
        qy = int(field.shape[-1] // 4)
        cropped_padded = field[
            :, :, qx : qx + field.shape[-2] // 2, qy : qy + field.shape[-1] // 2
        ]
    else:
        cx = int(field.shape[-2] // 2)
        cy = int(field.shape[-1] // 2)
        hx = int(size[-2] // 2)
        hy = int(size[-1] // 2)
        cropped_padded = field[:, :, cx - hx : cx + hx, cy - hy : cy + hy]
    
    if not channels_first:
        if len(orig_resolution) == 3:
            cropped_padded = cropped_padded.squeeze(0)
            cropped_padded = cropped_padded.permute(1, 2, 0)
        elif len(orig_resolution) == 4:
            cropped_padded = cropped_padded.permute(0, 2, 3, 1)
    
    if len(orig_resolution) == 2:
        cropped = cropped_padded.squeeze(0).squeeze(0)
    elif len(orig_resolution) == 3 and channels_first:
        cropped = cropped_padded.squeeze(0)
    elif len(orig_resolution) == 4 and channels_first:
        if orig_resolution[0] == 1:
            cropped = cropped_padded.squeeze(0)
        else:
            cropped = cropped_padded
    else:
        cropped = cropped_padded
    return cropped


def convolve2d(field, kernel):
    """
    Convolve a field with a kernel using frequency domain multiplication.
    
    This function performs 2D convolution by transforming both the field and kernel 
    to frequency domain, multiplying them, and transforming back to spatial domain.
    
    Parameters
    ----------
    field       : torch.tensor
                  Input field with MxN shape.
    kernel      : torch.tensor
                  Input kernel with MxN shape.

    Returns
    ----------
    convolved_field   : torch.tensor
                        Convolved field.
    """
    fr = torch.fft.fft2(field)
    fr2 = torch.fft.fft2(torch.flip(torch.flip(kernel, [1, 0]), [0, 1]))
    m, n = fr.shape
    convolved_field = torch.real(torch.fft.ifft2(fr * fr2))
    convolved_field = torch.roll(convolved_field, shifts=(int(n / 2 + 1), 0), dims=(1, 0))
    convolved_field = torch.roll(convolved_field, shifts=(int(m / 2 + 1), 0), dims=(0, 1))
    return convolved_field


def generate_2d_gaussian(
    kernel_length=[21, 21], nsigma=[3, 3], mu=[0, 0], normalize=False
):
    """
    Generate 2D Gaussian kernel.
    
    This function creates a 2D Gaussian kernel with specified dimensions and parameters.
    Inspired from https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    
    Parameters
    ----------
    kernel_length : list
                    Length of the Gaussian kernel along X and Y axes.
    nsigma        : list
                    Sigma of the Gaussian kernel along X and Y axes.
    mu            : list
                    Mu of the Gaussian kernel along X and Y axes.
    normalize     : bool
                    If set True, normalize the output to maximum value of 1.

    Returns
    ----------
    kernel_2d     : torch.tensor
                    Generated Gaussian kernel.
    """
    x = torch.linspace(
        -kernel_length[0] / 2.0, kernel_length[0] / 2.0, kernel_length[0]
    )
    y = torch.linspace(
        -kernel_length[1] / 2.0, kernel_length[1] / 2.0, kernel_length[1]
    )
    X, Y = torch.meshgrid(x, y, indexing="ij")
    if nsigma[0] == 0:
        nsigma[0] = 1e-5
    if nsigma[1] == 0:
        nsigma[1] = 1e-5
    kernel_2d = (
        1.0
        / (2.0 * torch.pi * nsigma[0] * nsigma[1])
        * torch.exp(
            -(
                (X - mu[0]) ** 2.0 / (2.0 * nsigma[0] ** 2.0)
                + (Y - mu[1]) ** 2.0 / (2.0 * nsigma[1] ** 2.0)
            )
        )
    )
    if normalize:
        kernel_2d = kernel_2d / kernel_2d.max()
    return kernel_2d


def generate_2d_dirac_delta(
    kernel_length=[21, 21], a=[3, 3], mu=[0, 0], theta=0, normalize=False
):
    """
    Generate 2D Dirac delta function using Gaussian approximation.
    
    This function creates a 2D Dirac delta function by using a Gaussian distribution 
    with very small standard deviations (a values) to approximate the behavior.
    Inspired from https://en.wikipedia.org/wiki/Dirac_delta_function
    
    Parameters
    ----------
    kernel_length : list
                    Length of the Dirac delta function along X and Y axes.
    a             : list
                    The scale factor in Gaussian distribution to approximate the Dirac delta function.
                    As a approaches zero, the Gaussian distribution becomes infinitely narrow and tall at the center (x=0), approaching the Dirac delta function.
    mu            : list
                    Mu of the Gaussian kernel along X and Y axes.
    theta         : float
                    The rotation angle of the 2D Dirac delta function.
    normalize     : bool
                    If set True, normalize the output to maximum value of 1.

    Returns
    ----------
    kernel_2d     : torch.tensor
                    Generated 2D Dirac delta function.
    """
    x = torch.linspace(
        -kernel_length[0] / 2.0, kernel_length[0] / 2.0, kernel_length[0]
    )
    y = torch.linspace(
        -kernel_length[1] / 2.0, kernel_length[1] / 2.0, kernel_length[1]
    )
    X, Y = torch.meshgrid(x, y, indexing="ij")
    X = X - mu[0]
    Y = Y - mu[1]
    theta = torch.as_tensor(theta)
    X_rot = X * torch.cos(theta) - Y * torch.sin(theta)
    Y_rot = X * torch.sin(theta) + Y * torch.cos(theta)
    kernel_2d = (1 / (abs(a[0] * a[1]) * torch.pi)) * torch.exp(
        -((X_rot / a[0]) ** 2 + (Y_rot / a[1]) ** 2)
    )
    if normalize:
        kernel_2d = kernel_2d / kernel_2d.max()
    return kernel_2d


def blur_gaussian(field, kernel_length=[21, 21], nsigma=[3, 3], padding="same"):
    """
    Blur a field using a Gaussian kernel.
    
    This function applies Gaussian blur to the input field using convolution with 
    a Gaussian kernel in the frequency domain.
    
    Parameters
    ----------
    field         : torch.tensor
                    MxN field to be blurred.
    kernel_length : list
                    Length of the Gaussian kernel along X and Y axes.
    nsigma        : list
                    Sigma of the Gaussian kernel along X and Y axes.
    padding       : int or string
                    Padding value, see torch.nn.functional.conv2d() for more.

    Returns
    ----------
    blurred_field : torch.tensor
                    Blurred field.
    """
    kernel = generate_2d_gaussian(kernel_length, nsigma).to(field.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    if len(field.shape) == 2:
        field = field.view(1, 1, field.shape[-2], field.shape[-1])
    blurred_field = torch.nn.functional.conv2d(field, kernel, padding="same")
    if field.shape[1] == 1:
        blurred_field = blurred_field.view(
            blurred_field.shape[-2], blurred_field.shape[-1]
        )
    return blurred_field


def correlation_2d(first_tensor, second_tensor):
    """
    Calculate the correlation between two tensors using FFT.
    
    This function computes the 2D correlation between two tensors using 
    frequency domain multiplication. It's equivalent to computing 
    cross-correlation using FFT techniques.
    
    Parameters
    ----------
    first_tensor  : torch.tensor
                    First tensor.
    second_tensor : torch.tensor
                    Second tensor.

    Returns
    ----------
    correlation   : torch.tensor
                    Correlation between the two tensors.
    """
    fft_first_tensor = torch.fft.fft2(first_tensor)
    fft_second_tensor = torch.fft.fft2(second_tensor)
    conjugate_second_tensor = torch.conj(fft_second_tensor)
    result = torch.fft.ifftshift(
        torch.fft.ifft2(fft_first_tensor * conjugate_second_tensor)
    )
    return result
