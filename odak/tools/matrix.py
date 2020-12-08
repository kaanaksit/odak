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

def generate_bandlimits(size=[512,512],levels=8):
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
        deltax = int(size[0]/masks.shape[0])
        deltay = int(size[1]/masks.shape[0])
        masks[
              i,
              cx-deltax/2*(i+1):cx+deltax/2*(i+1),
              cy-deltay/2*(i+1):cy+deltay/2*(i+1)
             ] = 1.
        masks[
              i,
              cx-deltax/2*i:cx+deltax/2*i,
              cy-deltay/2*i:cy+deltay/2*i
             ] = 0.
    masks = np.asarray(masks)
    return masks
