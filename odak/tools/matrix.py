from odak import np
import pkg_resources
import finufft

def nuifft2(field,fx,fy,sign=1,eps=10**(-12)):
    """
    A definition to take 2D Inverse Non-Uniform Fast Fourier Transform (NUFFT).

    Parameters
    ----------
    field       : ndarray
                  Input field.
    fx          : ndarray
                  Frequencies along x axis.
    fy          : ndarray
                  Frequencies along y axis.
    sign        : float
                  Sign of the exponential used in NUFFT kernel.
    eps         : float
                  Accuracy of NUFFT.

    Returns
    ----------
    result      : ndarray
                  Inverse NUFFT of the input field.
    """
    if np.__name__ == 'cupy':
        fx    = np.asnumpy(fx).astype(np.float64)
        fy    = np.asnumpy(fy).astype(np.float64)
        image = np.asnumpy(np.copy(field)).astype(np.complex128)
    else:
        image = np.copy(field).astype(np.complex128)
    result = finufft.nufft2d2(fx.flatten(),fy.flatten(),image,eps=eps,isign=sign)
    result = result.reshape(field.shape)
    if np.__name__ == 'cupy':
        result = np.asarray(result)
    return result

def nufft2(field,fx,fy,size=None,sign=1,eps=10**(-12)):
    """
    A definition to take 2D Non-Uniform Fast Fourier Transform (NUFFT).

    Parameters
    ----------
    field       : ndarray
                  Input field.
    fx          : ndarray
                  Frequencies along x axis.
    fy          : ndarray
                  Frequencies along y axis.
    size        : list or ndarray
                  Shape of the NUFFT calculated for an input field.
    sign        : float
                  Sign of the exponential used in NUFFT kernel.
    eps         : float
                  Accuracy of NUFFT.

    Returns
    ----------
    result      : ndarray
                  NUFFT of the input field.
    """
    if np.__name__ == 'cupy':
        fx    = np.asnumpy(fx).astype(np.float64)
        fy    = np.asnumpy(fy).astype(np.float64)
        image = np.asnumpy(np.copy(field)).astype(np.complex128)
    else:
        image = np.copy(field).astype(np.complex128)
    if type(size) == type(None):
        result = finufft.nufft2d1(
                                  fx.flatten(),
                                  fy.flatten(),
                                  image.flatten(),
                                  image.shape,
                                  eps=eps,
                                  isign=sign
                                 )
    else:
        result = finufft.nufft2d1(
                                  fx.flatten(),
                                  fy.flatten(),
                                  image.flatten(),
                                  (size[0],size[1]),
                                  eps=eps,
                                  isign=sign
                                 )
    if np.__name__ == 'cupy':
        result = np.asarray(result)
    return result

def generate_bandlimits(size=[512,512],levels=9):
    """
    A definition to calculate octaves used in bandlimiting frequencies in the frequency domain.

    Parameters
    ----------
    size       : list
                 Size of each mask in octaves.

    Returns
    ----------
    masks      : ndarray
                 Masks (Octaves).
    """
    masks = np.zeros((levels,size[0],size[1]))
    cx     = size[0]/2
    cy     = size[1]/2
    for i in range(0,masks.shape[0]):
        deltax = int((size[0])/(2**(i+1)))
        deltay = int((size[1])/(2**(i+1)))
        masks[
              i,
              cx-deltax:cx+deltax,
              cy-deltay:cy+deltay
             ] = 1.
        masks[
              i,
              cx-deltax/2.:cx+deltax/2.,
              cy-deltay/2.:cy+deltay/2.
             ] = 0.
    masks = np.asarray(masks)
    return masks

def zero_pad(field,size=None):
    """
    Definition to zero pad a MxN array to 2Mx2N array.

    Parameters
    ----------
    field             : ndarray
                        Input field MxN array.

    Returns
    ----------
    field_zero_padded : ndarray
                        Zeropadded version of the input field.
    """
    if type(size) == type(None):
        hx = int(np.ceil(field.shape[0])/2)
        hy = int(np.ceil(field.shape[1])/2)
    else:
        hx = int(np.ceil((size[0]-field.shape[0])/2))
        hy = int(np.ceil((size[1]-field.shape[1])/2))
    field_zero_padded = np.pad(field,([hx,hx],[hy,hy]), constant_values=(0,0))
    return field_zero_padded

def crop_center(field,size=None):
    """
    Definition to crop the center of a field with 2Mx2N size. The outcome is a MxN array.

    Parameters
    ----------
    field       : ndarray
                  Input field 2Mx2N array.

    Returns
    ----------
    cropped     : ndarray
                  Cropped version of the input field.
    """
    if type(size) == type(None):
        qx      = int(np.ceil(field.shape[0])/4)
        qy      = int(np.ceil(field.shape[1])/4)
        cropped = np.copy(field[qx:3*qx,qy:3*qy])
    else:
        cx      = int(np.ceil(field.shape[0]/2))
        cy      = int(np.ceil(field.shape[1]/2))
        hx      = int(np.ceil(size[0]/2))
        hy      = int(np.ceil(size[1]/2))
        print(cx,cy,field.shape)
        cropped = np.copy(field[cx-hx:cx+hx,cy-hy:cy+hy]) 
    return cropped
