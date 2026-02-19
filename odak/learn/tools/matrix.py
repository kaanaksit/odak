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


def zero_pad(field, size=None, method="center"):
    """
    Zero pad a field to double its size or specified size.
    
    This function pads a field with zeros to either double its size (default) 
    or to a specified size. The input can be 2D, 3D or 4D tensors.
    
    Parameters
    ----------
    field             : torch.tensor
                        Input field MxN or KxJxMxN or KxMxNxJ array.
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
    if len(field.shape) < 3:
        field = field.unsqueeze(0)
    if len(field.shape) < 4:
        field = field.unsqueeze(0)
    permute_flag = False
    if field.shape[-1] < 5:
        permute_flag = True
        field = field.permute(0, 3, 1, 2)
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
    if permute_flag:
        field_zero_padded = field_zero_padded.permute(0, 2, 3, 1)
    if len(orig_resolution) == 2:
        field_zero_padded = field_zero_padded.squeeze(0).squeeze(0)
    if len(orig_resolution) == 3:
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
                  Input field 2M x 2N or K x L x 2M x 2N or K x 2M x 2N x L array.
    size        : list
                  Dimensions to crop with respect to center of the image (e.g., M x N or 1 x 1 x M x N).
                  If None, crops to half of the current size.

    Returns
    ----------
    cropped     : torch.tensor
                  Cropped version of the input field.
    """
    orig_resolution = field.shape
    if len(field.shape) < 3:
        field = field.unsqueeze(0)
    if len(field.shape) < 4:
        field = field.unsqueeze(0)
    permute_flag = False
    if field.shape[-1] < 5:
        permute_flag = True
        field = field.permute(0, 3, 1, 2)
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
    cropped = cropped_padded
    if permute_flag:
        cropped = cropped.permute(0, 2, 3, 1)
    if len(orig_resolution) == 2:
        cropped = cropped_padded.squeeze(0).squeeze(0)
    if len(orig_resolution) == 3:
        cropped = cropped_padded.squeeze(0)
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
