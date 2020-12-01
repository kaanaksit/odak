from odak import np
import finufft
import pkg_resources

def nufft2(field,fx,fy,sx,sy,sign=1):
    """
    """
    if np.__name__ == 'cupy':
        fx    = np.asnumpy(fx)
        fy    = np.asnumpy(fy)
        sx    = np.asnumpy(sx)
        sy    = np.asnumpy(sy)
        image = np.asnumpy(np.copy(field)).astype(np.complex128)
    else:
        image = np.copy(field).astype(np.complex128)
    result = finufft.nufft2d3(
                              fx.flatten(), 
                              fy.flatten(), 
                              image.flatten(),
                              sx.flatten(),
                              sy.flatten(),
                              isign=sign
                             )
    if np.__name__ == 'cupy':
        result = np.asarray(result)
    return result
