from odak import np
import torch,torch.nn

def quantize(image_field,bits=4):
    """ 
    Definition to quantize a image field (0-255, 8 bit) to a certain bits level.

    Parameters
    ----------
    image_field : torch.tensor
                  Input image field.
    bits        : int
                  A value in between 0 to 8. Can not be zero.

    Returns
    ----------
    new_field   : torch.tensor
                  Quantized image field.
    """
    divider   = 2**(8-bits)
    new_field = image_field/divider
    new_field = new_field.int()
    return new_field

def zero_pad(field,size=None,method='center'):
    """
    Definition to zero pad a MxN array to 2Mx2N array.

    Parameters
    ----------
    field             : ndarray
                        Input field MxN array.
    size              : list
                        Size to be zeropadded.
    method            : str
                        Zeropad either by placing the content to center or to the left.

    Returns
    ----------
    field_zero_padded : ndarray
                        Zeropadded version of the input field.
    """
    if type(size) == type(None):
        hx = int(torch.ceil(torch.tensor([field.shape[0]/2])))
        hy = int(torch.ceil(torch.tensor([field.shape[1]/2])))
    else:
        hx = int(torch.ceil(torch.tensor([(size[0]-field.shape[0])/2])))
        hy = int(torch.ceil(torch.tensor([(size[1]-field.shape[1])/2])))
    if method == 'center':
        m = torch.nn.ZeroPad2d((hy, hy, hx, hx))
    elif method == 'left aligned':
        m = torch.nn.ZeroPad2d((0, hy*2, 0, hx*2))
    field_zero_padded = m(field)
    if type(size) != type(None):
        field_zero_padded = field_zero_padded[0:size[0],0:size[1]]
    return field_zero_padded

def crop_center(field,size=None):
    """
    Definition to crop the center of a field with 2Mx2N size. The outcome is a MxN array.

    Parameters
    ----------
    field       : ndarray
                  Input field 2Mx2N array.
    size        : list
                  Dimensions to crop with respect to center of the image.

    Returns
    ----------
    cropped     : ndarray
                  Cropped version of the input field.
    """
    if type(size) == type(None):
        qx      = int(torch.ceil(torch.tensor(field.shape[0])/4))
        qy      = int(torch.ceil(torch.tensor(field.shape[1])/4))
        cropped = field[qx:3*qx,qy:3*qy]
    else:
        cx      = int(torch.ceil(torch.tensor(field.shape[0]/2)))
        cy      = int(torch.ceil(torch.tensor(field.shape[1]/2)))
        hx      = int(torch.ceil(torch.tensor(size[0]/2)))
        hy      = int(torch.ceil(torch.tensor(size[1]/2)))
        cropped = field[cx-hx:cx+hx,cy-hy:cy+hy]
    return cropped

def convolve2d(field,kernel):
    """
    Definition to convolve a field with a kernel by multiplying in frequency space.

    Parameters
    ----------
    field       : torch.tensor
                  Input field with MxN shape.
    kernel      : torch.tensor
                  Input kernel with MxN shape.

    Returns
    ----------
    new_field   : torch.tensor
                  Convolved field.
    """
    fr        = torch.fft.fft2(field)
    fr2       = torch.fft.fft2(torch.flip(torch.flip(kernel,[1,0]),[0,1]))
    m,n       = fr.shape
    new_field = torch.real(torch.fft.ifft2(fr*fr2))
    new_field = torch.roll(new_field,shifts=(int(n/2+1),0),dims=(1,0))
    new_field = torch.roll(new_field,shifts=(int(m/2+1),0),dims=(0,1))
    return new_field

def generate_2d_gaussian(kernel_length=[21,21], nsigma=[3,3]):
    """
    Generate 2D Gaussian kernel. Inspired from https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy

    Parameters
    ----------
    kernel_length : list
                    Length of the Gaussian kernel along X and Y axes.
    nsigma        : list
                    Sigma of the Gaussian kernel along X and Y axes.

    Returns
    ----------
    kernel_2d     : torch.tensor
                    Generated Gaussian kernel.
    """
    x           = torch.linspace(-nsigma[0], nsigma[0], kernel_length[0]+1)
    y           = torch.linspace(-nsigma[1], nsigma[1], kernel_length[1]+1)
    xx, yy      = torch.meshgrid(x, y)
    nsigma      = torch.tensor(nsigma)
    kernel_2d   = torch.exp(-0.5*(torch.square(xx)/torch.square(nsigma[0]) + torch.square(yy)/torch.square(nsigma[1])))
    kernel_2d   = kernel_2d/kernel_2d.sum()
    return kernel_2d

def blur_gaussian(field,kernel_length=[21,21],nsigma=[3,3]):
    """
    A definition to blur a field using a Gaussian kernel.

    Parameters
    ----------
    field         : ndarray
                    MxN field.
    kernel_length : list
                    Length of the Gaussian kernel along X and Y axes.
    nsigma        : list
                    Sigma of the Gaussian kernel along X and Y axes.

    Returns
    ----------
    blurred_field : ndarray
                    Blurred field.
    """
    kernel        = generate_2d_gaussian(kernel_length,nsigma).to(field.device)
    kernel        = zero_pad(kernel,field.shape)
    blurred_field = convolve2d(field,kernel)
    blurred_field = blurred_field/torch.amax(blurred_field)
    return blurred_field
