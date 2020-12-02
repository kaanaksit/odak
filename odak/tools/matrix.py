from odak import np
import finufft
import pkg_resources

def nufft2(field,fx,fy,sign=1,eps=10**(-12)):
    """
    """
    if np.__name__ == 'cupy':
        fx    = np.asnumpy(fx)
        fy    = np.asnumpy(fy)
        image = np.asnumpy(np.copy(field)).astype(np.complex128)
    else:
        image = np.copy(field).astype(np.complex128)
    result = finufft.nufft2d1(
                              fx.flatten(), 
                              fy.flatten(), 
                              image.flatten(),
                              image.shape,
                              isign=sign,
                              eps=eps
                             )
    result = result.reshape(field.shape)
    if np.__name__ == 'cupy':
        result = np.asarray(result)
    return result
