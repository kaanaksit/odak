from odak import np
import pkg_resources
import finufft

def nuifft2(field,fx,fy,sign=1,eps=10**(-12)):
    """
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
