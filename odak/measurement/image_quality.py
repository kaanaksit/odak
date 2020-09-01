from odak import np

def modulation_transfer_function(img,px_size,fit_degree=[10,10]):
    """
    Definition to compute modulation transfer function. This definition is based on the work by Peter Burns. For more consult Burns, Peter D. "Slanted-edge MTF for digital camera and scanner analysis." Is and Ts Pics Conference. SOCIETY FOR IMAGING SCIENCE & TECHNOLOGY, 2000.

    Parameters
    ----------
    img          : ndarray
                   Region of interest provided from a complete image that contains a slanted edge.
    px_size      : ndarray
                   Physical angular sizes that each pixels corresponds to on the image plane both for X and Y axes.
    fit_degree   : list
                   Degrees for polynomial fits in both X and Y axes.

    Returns
    ----------
    mtf          : ndarray
                   Calculated modulation transfer function along X and Y axes.
    frq          : ndarray
                   Frequencies of the calculated MTF.
    p            : numpy.poly1d
                   Polynomial fits for MTF along X and Y axes.
    """
    img_m_x                    = img[:,int(img.shape[1]/2)]
    img_m_y                    = img[int(img.shape[0]/2),:]    
    # 1st derivative of both values: "Line Spread Function",
    der_x                      = np.gradient(img_m_x)
    der_y                      = np.gradient(img_m_y)
    # Fourier transform of the first derivative: "Modulation Transfer Function",
    mtf_x                      = np.fft.fft(der_x) #/len(der_x)
    mtf_x                     /= np.amax(mtf_x)
    mtf_x                      = mtf_x[range(int(len(der_x)/2))]
    mtf_y                      = np.fft.fft(der_y) #/len(der_y)
    mtf_y                     /= np.amax(mtf_y)
    mtf_y                      = mtf_y[range(int(len(der_y)/2))]
    # Arrange the corresponding frequencies,
    n_x                        = len(der_x) # length of the signal,
    k_x                        = np.arange(n_x)
    T_x                        = n_x*px_size[0]
    frq_x                      = k_x/T_x
    frq_x                      = frq_x[range(int(n_x/2))]
    n_y                        = len(der_y) # length of the signal,
    k_y                        = np.arange(n_y)
    T_y                        = n_y*px_size[1]
    frq_y                      = k_y/T_y
    frq_y                      = frq_y[range(int(n_y/2))]
    # Polyfit for MTFs.
    fun_poly_x                 = np.polyfit(frq_x,abs(mtf_x),fit_degree[0])
    fun_poly_y                 = np.polyfit(frq_y,abs(mtf_y),fit_degree[1])
    p_x                        = np.poly1d(fun_poly_x)
    p_y                        = np.poly1d(fun_poly_y)
    return [mtf_x,mtf_y],[frq_x,frq_y],[p_x,p_y]
