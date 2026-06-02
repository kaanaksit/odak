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


def smooth_pad(field, size=None, method="center", smooth_factor=None):
    """
    Smooth pad a field to double its size or specified size with smooth transition (apodization).
    
    This function implements a Tukey window (cosine-tapered window) to pad a field with a
    smooth transition from the content edge to zero, avoiding sharp edges.
    The original content area is preserved at full value.
    The transition rate is controllable along X and Y axes.
    If transition rate is not provided, a default transition is applied across
    the entire padding region, reaching zero at the canvas boundary.
    
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
                        If None, a default falloff covers the entire padding region.
    
    Returns
    ----------
    field_smooth_padded : torch.tensor
                          Smoothpadded version of the input field with gradual falloff.
    
    Examples
    --------
    >>> import torch
    >>> from odak.learn.tools.matrix import smooth_pad
    >>> 
    >>> # Create a 100x100 field
    >>> field = torch.ones(100, 100)
    >>> 
    >>> # Double the size with default smooth factor
    >>> padded = smooth_pad(field)  # 200x200 output
    >>> 
    >>> # Pad to specific size with anisotropic falloff
    >>> padded = smooth_pad(field, size=[200, 300], smooth_factor=[0.2, 0.6])
    >>> # 200x300 output with gentle X-axis fade (0.2) and steep Y-axis fade (0.6)
    >>> 
    >>> # Place content at top-left corner
    >>> padded = smooth_pad(field, size=[200, 200], method="left")
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
        if not channels_first:
            field = field.permute(2, 0, 1)
    elif orig_ndim == 4:
        if not channels_first:
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

    # Determine padding size, field resolution, smooth factors, datatype
    h, w = resolution[-2], resolution[-1]
    field_h, field_w = field.shape[-2], field.shape[-1]
    sx, sy = smooth_factor if smooth_factor is not None else (None, None)
    real_dtype = field.real.dtype if field.is_complex() else field.dtype

    if h < field_h or w < field_w:
        raise ValueError(f"Target size ({h}, {w}) is smaller than field size ({field_h}, {field_w}).")

    if method == "center":
        pad_h_top = (h - field_h) // 2
        pad_h_bot = h - field_h - pad_h_top
        pad_w_left = (w - field_w) // 2
        pad_w_right = w - field_w - pad_w_left
    elif method == "left":
        pad_h_top, pad_w_left = 0, 0
        pad_h_bot = h - field_h
        pad_w_right = w - field_w
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'center' or 'left'.")

    def _rep_pad(t):
        return torch.nn.functional.pad(t, (pad_w_left, pad_w_right, pad_h_top, pad_h_bot), mode='replicate')

    # Replicated padding region is detached from the computation graph,
    # preventing gradient accumulation/amplification at border pixels.
    with torch.no_grad():
        if field.is_complex():
            rep = torch.complex(_rep_pad(field.real), _rep_pad(field.imag))
        else:
            rep = _rep_pad(field)

    def _zero_pad(t):
        return torch.nn.functional.pad(t, (pad_w_left, pad_w_right, pad_h_top, pad_h_bot))

    if field.is_complex():
        zero = torch.complex(_zero_pad(field.real), _zero_pad(field.imag))
    else:
        zero = _zero_pad(field)

    # Combine zero padding (with gradients) and replicate padding (without gradients) using a content mask
    content_mask = torch.zeros(field.shape[0], field.shape[1], h, w, device=field.device, dtype=torch.bool)
    content_mask[:, :, pad_h_top:pad_h_top + field_h, pad_w_left:pad_w_left + field_w] = True
    padded = torch.where(content_mask, zero, rep)

    # Construct window function
    def _taper_1d(total: int, left_start: int, size: int, factor=None) -> torch.Tensor:
        device = field.device
        window = torch.zeros(total, device=device, dtype=torch.float32)
        window[left_start: left_start + size] = 1.0

        if factor is None:
            # Auto mode: cosine taper across full padding
            if left_start > 0:
                t = torch.arange(left_start, device=device, dtype=torch.float32)
                window[:left_start] = 0.5 * (1.0 - torch.cos(torch.pi * t / max(left_start - 1, 1)))
            right_start = left_start + size
            right_length = total - right_start
            if right_length > 0:
                t = torch.arange(right_length, device=device, dtype=torch.float32)
                window[right_start:] = 0.5 * (1.0 + torch.cos(torch.pi * t / max(right_length - 1, 1)))
        else:
            # Factor mode: power-cosine taper
            effective_exp = 100.0 ** max(factor, 0.0)
            if left_start > 0:
                t = torch.arange(left_start, device=device, dtype=torch.float32)
                t_norm = (left_start - 1 - t) / max(left_start - 1, 1)
                window[:left_start] = torch.cos(torch.pi * t_norm / 2).clamp(min=0) ** effective_exp
            right_start = left_start + size
            right_length = total - right_start
            if right_length > 0:
                t = torch.arange(right_length, device=device, dtype=torch.float32)
                t_norm = t / max(right_length - 1, 1)
                window[right_start:] = torch.cos(torch.pi * t_norm / 2).clamp(min=0) ** effective_exp
        return window.to(real_dtype)

    # Apply window function to the padded field
    wy = _taper_1d(h, pad_h_top, field_h, sy)
    wx = _taper_1d(w, pad_w_left, field_w, sx)
    wxy = torch.outer(wy, wx).unsqueeze(0).unsqueeze(0)      # (1, 1, h, w)
    result = padded * wxy                                    # (B, C, h, w) × (1, 1, h, w)

    if not channels_first:
        if len(orig_resolution) == 3:
            result = result.squeeze(0)
            result = result.permute(1, 2, 0)
        elif len(orig_resolution) == 4:
            result = result.permute(0, 2, 3, 1)

    if len(orig_resolution) == 2:
        result = result.squeeze(0).squeeze(0)
    elif len(orig_resolution) == 3 and channels_first:
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
    
    Examples
    --------
    >>> import torch
    >>> from odak.learn.tools.matrix import zero_pad
    >>> 
    >>> # Create a 100x100 field
    >>> field = torch.ones(100, 100)
    >>> 
    >>> # Double the size (default)
    >>> padded = zero_pad(field)  # 200x200 output with content centered
    >>> 
    >>> # Pad to specific size
    >>> padded = zero_pad(field, size=[200, 300])  # 200x300 output
    >>> 
    >>> # Place content at top-left corner
    >>> padded = zero_pad(field, size=[200, 200], method="left")
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
    
    Examples
    --------
    >>> import torch
    >>> from odak.learn.tools.matrix import crop_center
    >>> 
    >>> # Create a 200x200 field
    >>> field = torch.ones(200, 200)
    >>> 
    >>> # Crop to half the size (default)
    >>> cropped = crop_center(field)  # 100x100 output
    >>> 
    >>> # Crop to specific size
    >>> cropped = crop_center(field, size=[50, 150])  # 50x150 output
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
