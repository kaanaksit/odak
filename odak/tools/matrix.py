from odak import np
import pkg_resources
#import nufft2d

def nufft2(field,fx,fy,a,sign=1,eps=10**(-12)):
    """
    """
    if np.__name__ == 'cupy':
        fx    = np.asnumpy(fx)
        fy    = np.asnumpy(fy)
        image = np.asnumpy(np.copy(field)).astype(np.complex128)
    else:
        image = np.copy(field).astype(np.complex128)
    result = nufft2d.nufft2d3f90(
                                 image.shape,
                                 fx,
                                 fy,
                                 image,
                                 sign,
                                 eps,
                                 image.shape,
                                 a
                                )
    if np.__name__ == 'cupy':
        result = np.asarray(result)
    return result
 
