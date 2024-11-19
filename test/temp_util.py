import torch
import numpy as np
import math
import torch.nn.functional as F

def get_z_vec(opt_depths, number_of_planes):
    device = opt_depths.device
    # Concatenate 1.0, opt_depths, and 0.0 into a single tensor
    z_vec = torch.cat([
        torch.tensor([1.0, ], dtype=torch.float32, device=device),
        opt_depths,
        torch.tensor([0.,], dtype=torch.float32, device=device)
    ])
    # Sort the tensor and flip it to get descending order
    z_vec, _ = torch.sort(z_vec, descending=False)
    return z_vec

def optical_MTF_Barten1999(u: torch.Tensor, sigma: torch.Tensor = torch.tensor(0.01)) -> torch.Tensor:
    return torch.exp(-2 * torch.pi**2 * sigma**2 * u**2)

def pupil_diameter_Barten1999(L: torch.Tensor, X_0: torch.Tensor = torch.tensor(60.0), Y_0: torch.Tensor = None) -> torch.Tensor:
    Y_0 = X_0 if Y_0 is None else Y_0
    return 5 - 3 * torch.tanh(0.4 * torch.log10(L * X_0 * Y_0 / 40**2))

def sigma_Barten1999(sigma_0: torch.Tensor = torch.tensor(0.5 / 60), C_ab: torch.Tensor = torch.tensor(0.08 / 60), d: torch.Tensor = torch.tensor(2.1)) -> torch.Tensor:
    return torch.hypot(sigma_0, C_ab * d)

def retinal_illuminance_Barten1999(L: torch.Tensor, d: torch.Tensor = torch.tensor(2.1), apply_stiles_crawford_effect_correction: bool = True) -> torch.Tensor:
    E = (torch.pi * d**2) / 4 * L
    if apply_stiles_crawford_effect_correction:
        E *= 1 - (d / 9.7) ** 2 + (d / 12.4) ** 4
    return E

def maximum_angular_size_Barten1999(u: torch.Tensor, X_0: torch.Tensor = torch.tensor(60.0), X_max: torch.Tensor = torch.tensor(12.0), N_max: torch.Tensor = torch.tensor(15.0)) -> torch.Tensor:
    return (1 / X_0**2 + 1 / X_max**2 + u**2 / N_max**2) ** -0.5

def contrast_sensitivity_function_Barten1999(
    u: torch.Tensor,
    sigma: torch.Tensor = None,
    k: torch.Tensor = torch.tensor(3.0),
    T: torch.Tensor = torch.tensor(0.1),
    X_0: torch.Tensor = torch.tensor(60.0),
    Y_0: torch.Tensor = None,
    X_max: torch.Tensor = torch.tensor(12.0),
    Y_max: torch.Tensor = None,
    N_max: torch.Tensor = torch.tensor(15.0),
    n: torch.Tensor = torch.tensor(0.03),
    p: torch.Tensor = torch.tensor(1.2274e6),
    E: torch.Tensor = None,
    phi_0: torch.Tensor = torch.tensor(3e-8),
    u_0: torch.Tensor = torch.tensor(7.0),
) -> torch.Tensor:
    Y_0 = X_0 if Y_0 is None else Y_0
    Y_max = X_max if Y_max is None else Y_max

    if sigma is None:
        d = pupil_diameter_Barten1999(torch.tensor(20.0), X_0, Y_0)
        sigma = sigma_Barten1999(torch.tensor(0.5 / 60), torch.tensor(0.08 / 60), d)

    if E is None:
        d = pupil_diameter_Barten1999(torch.tensor(20.0), X_0, Y_0)
        E = retinal_illuminance_Barten1999(torch.tensor(20.0), d)

    M_opt = optical_MTF_Barten1999(u, sigma)
    M_as = 1 / (
        maximum_angular_size_Barten1999(u, X_0, X_max, N_max)
        * maximum_angular_size_Barten1999(u, Y_0, Y_max, N_max)
    )

    S = (M_opt / k) / torch.sqrt(
        2
        / T
        * M_as
        * (1 / (n * p * E) + phi_0 / (1 - torch.exp(-((u / u_0) ** 2))))
    )

    return S

def max_min_normalization(input, type="torch"):
    if type.lower() == "numpy":
        return (input - np.min(input)) / (np.max(input) - np.min(input)) 
    elif type.lower() == "torch":
        return (input - torch.min(input)) / (torch.max(input) - torch.min(input)) 
    
### Pyramid
def ceildiv(a, b):
    return -(-a // b)

# Decimated Laplacian pyramid
class lpyr_dec():

    def __init__(self, W, H, ppd, device):
        self.device = device
        self.ppd = ppd
        self.min_freq = 0.2
        self.W = W
        self.H = H

        max_levels = int(np.floor(np.log2(min(self.H, self.W))))-1

        bands = np.concatenate([[1.0], np.power(2.0, -np.arange(0.0,14.0)) * 0.3228], 0) * self.ppd/2.0 

        # print(max_levels)
        # print(bands)
        # sys.exit(0)

        invalid_bands = np.array(np.nonzero(bands <= self.min_freq)) # we want to find first non0, length is index+1

        if invalid_bands.shape[-2] == 0:
            max_band = max_levels
        else:
            max_band = invalid_bands[0][0]

        # max_band+1 below converts index into count
        self.height = np.clip(max_band+1, 0, max_levels) # int(np.clip(max(np.ceil(np.log2(ppd)), 1.0)))
        self.band_freqs = np.array([1.0] + [0.3228 * 2.0 **(-f) for f in range(self.height)]) * self.ppd/2.0

        self.pyr_shape = self.height * [None] # shape (W,H) of each level of the pyramid
        self.pyr_ind = self.height * [None]   # index to the elements at each level

        cH = H
        cW = W
        for ll in range(self.height):
            self.pyr_shape[ll] = (cH, cW)
            cH = ceildiv(H,2)
            cW = ceildiv(W,2)

    def get_freqs(self):
        return self.band_freqs

    def get_band_count(self):
        return self.height+1

    def get_band(self, bands, band):
        if band == 0 or band == (len(bands)-1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        return bands[band] * band_mul

    def set_band(self, bands, band, data):
        if band == 0 or band == (len(bands)-1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        bands[band] = data / band_mul

    def get_gband(self, gbands, band):
        return gbands[band]

    # def get_gband_count(self):
    #     return self.height #len(gbands)

    # def clear(self):
    #     for pyramid in self.P:
    #         for level in pyramid:
    #             # print ("deleting " + str(level))
    #             del level

    def decompose(self, image): 
        # assert len(image.shape)==4, "NCHW (C==1) is expected, got " + str(image.shape)
        # assert image.shape[-2] == self.H
        # assert image.shape[-1] == self.W

        # self.image = image

        return self.laplacian_pyramid_dec(image, self.height+1)

    def reconstruct(self, bands):
        img = bands[-1]

        for i in reversed(range(0, len(bands)-1)):
            img = self.gausspyr_expand(img, [bands[i].shape[-2], bands[i].shape[-1]])
            img += bands[i]

        return img

    def laplacian_pyramid_dec(self, image, levels = -1, kernel_a = 0.4):
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return []

        lpyr = []
        for i in range(height-1):
            layer = gpyr[i] - self.gausspyr_expand(gpyr[i+1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
            lpyr.append(layer)

        lpyr.append(gpyr[height-1])

        # print("laplacian pyramid summary:")
        # print("self.height = %d" % self.height)
        # print("height      = %d" % height)
        # print("len(lpyr)   = %d" % len(lpyr))
        # print("len(gpyr)   = %d" % len(gpyr))
        # sys.exit(0)

        return lpyr, gpyr

    def interleave_zeros_and_pad(self, x, exp_size, dim):
        new_shape = [*x.shape]
        new_shape[dim] = exp_size[dim]+4
        z = torch.zeros( new_shape, dtype=x.dtype, device=x.device)
        odd_no = (exp_size[dim]%2)
        if dim==-2:
            z[:,:,2:-2:2,:] = x
            z[:,:,0,:] = x[:,:,0,:]
            z[:,:,-2+odd_no,:] = x[:,:,-1,:]
        elif dim==-1:
            z[:,:,:,2:-2:2] = x
            z[:,:,:,0] = x[:,:,:,0]
            z[:,:,:,-2+odd_no] = x[:,:,:,-1]
        else:
            assert False, "Wrong dimension"

        return z

    def gaussian_pyramid_dec(self, image, levels = -1, kernel_a = 0.4):

        default_levels = int(np.floor(np.log2(min(image.shape[-2], image.shape[-1]))))

        if levels == -1:
            levels = default_levels
        if levels > default_levels:
            raise Exception("Too many levels (%d) requested. Max is %d for %s" % (levels, default_levels, image.shape))

        res = [image]

        for i in range(1, levels):
            res.append(self.gausspyr_reduce(res[i-1], kernel_a))

        return res


    def sympad(self, x, padding, axis):
        if padding == 0:
            return x
        else:
            beg = torch.flip(torch.narrow(x, axis, 0,        padding), [axis])
            end = torch.flip(torch.narrow(x, axis, -padding, padding), [axis])

            return torch.cat((beg, x, end), axis)

    def get_kernels( self, im, kernel_a = 0.4 ):

        ch_dim = len(im.shape)-2
        if hasattr(self, "K_horiz") and ch_dim==self.K_ch_dim:
            return self.K_vert, self.K_horiz

        K = torch.tensor([0.25 - kernel_a/2.0, 0.25, kernel_a, 0.25, 0.25 - kernel_a/2.0], device=im.device, dtype=im.dtype)
        self.K_vert = torch.reshape(K, (1,)*ch_dim + (K.shape[0], 1))
        self.K_horiz = torch.reshape(K, (1,)*ch_dim + (1, K.shape[0]))
        self.K_ch_dim = ch_dim
        return self.K_vert, self.K_horiz
        

    def gausspyr_reduce(self, x, kernel_a = 0.4):

        K_vert, K_horiz = self.get_kernels( x, kernel_a )

        B, C, H, W = x.shape
        y_a = F.conv2d(x.view(-1,1,H,W), K_vert, stride=(2,1), padding=(2,0)).view(B,C,-1,W)

        # Symmetric padding 
        y_a[:,:,0,:] += x[:,:,0,:]*K_vert[0,0,1,0] + x[:,:,1,:]*K_vert[0,0,0,0]
        if (x.shape[-2] % 2)==1: # odd number of rows
            y_a[:,:,-1,:] += x[:,:,-1,:]*K_vert[0,0,3,0] + x[:,:,-2,:]*K_vert[0,0,4,0]
        else: # even number of rows
            y_a[:,:,-1,:] += x[:,:,-1,:]*K_vert[0,0,4,0]

        H = y_a.shape[-2]
        y = F.conv2d(y_a.view(-1,1,H,W), K_horiz, stride=(1,2), padding=(0,2)).view(B,C,H,-1)

        # Symmetric padding 
        y[:,:,:,0] += y_a[:,:,:,0]*K_horiz[0,0,0,1] + y_a[:,:,:,1]*K_horiz[0,0,0,0]
        if (x.shape[-2] % 2)==1: # odd number of columns
            y[:,:,:,-1] += y_a[:,:,:,-1]*K_horiz[0,0,0,3] + y_a[:,:,:,-2]*K_horiz[0,0,0,4]
        else: # even number of columns
            y[:,:,:,-1] += y_a[:,:,:,-1]*K_horiz[0,0,0,4] 

        return y

    def gausspyr_expand_pad(self, x, padding, axis):
        if padding == 0:
            return x
        else:
            beg = torch.narrow(x, axis, 0,        padding)
            end = torch.narrow(x, axis, -padding, padding)

            return torch.cat((beg, x, end), axis)

    # This function is (a bit) faster
    def gausspyr_expand(self, x, sz = None, kernel_a = 0.4):
        if sz is None:
            sz = [x.shape[-2]*2, x.shape[-1]*2]

        K_vert, K_horiz = self.get_kernels( x, kernel_a )

        y_a = self.interleave_zeros_and_pad(x, dim=-2, exp_size=sz)

        B, C, H, W = y_a.shape
        y_a = F.conv2d(y_a.view(-1,1,H,W), K_vert*2).view(B,C,-1,W)

        y   = self.interleave_zeros_and_pad(y_a, dim=-1, exp_size=sz)
        B, C, H, W = y.shape

        y   = F.conv2d(y.view(-1,1,H,W), K_horiz*2).view(B,C,H,-1)

        return y

    def interleave_zeros(self, x, dim):
        z = torch.zeros_like(x, device=self.device)
        if dim==2:
            return torch.cat([x,z],dim=3).view(x.shape[0], x.shape[1], 2*x.shape[2],x.shape[3])
        elif dim==3:
            return torch.cat([x.permute(0,1,3,2),z.permute(0,1,3,2)],dim=3).view(x.shape[0], x.shape[1], 2*x.shape[3],x.shape[2]).permute(0,1,3,2)



# Decimated Laplacian pyramid with a bit better interface - stores all bands within the object
class lpyr_dec_2(lpyr_dec):

    def __init__(self, W, H, ppd, device, keep_gaussian=False):
        self.device = device
        self.ppd = ppd
        self.min_freq = 0.2
        self.W = W
        self.H = H
        self.keep_gaussian=keep_gaussian

        max_levels = int(np.floor(np.log2(min(self.H, self.W))))-1

        bands = np.concatenate([[1.0], np.power(2.0, -np.arange(0.0,14.0)) * 0.3228], 0) * self.ppd/2.0 

        # print(max_levels)
        # print(bands)
        # sys.exit(0)

        invalid_bands = np.array(np.nonzero(bands <= self.min_freq)) # we want to find first non0, length is index+1

        if invalid_bands.shape[-2] == 0:
            max_band = max_levels
        else:
            max_band = invalid_bands[0][0]

        # max_band+1 below converts index into count
        self.height = np.clip(max_band+1, 0, max_levels) # int(np.clip(max(np.ceil(np.log2(ppd)), 1.0)))
        self.band_freqs = np.array([1.0] + [0.3228 * 2.0 **(-f) for f in range(self.height)]) * self.ppd/2.0

        self.pyr_shape = self.height * [None] # shape (W,H) of each level of the pyramid
        self.pyr_ind = self.height * [None]   # index to the elements at each level

        cH = H
        cW = W
        for ll in range(self.height):
            self.pyr_shape[ll] = (cH, cW)
            cH = ceildiv(H,2)
            cW = ceildiv(W,2)

        self.lbands = [None] * (self.height+1) # Laplacian pyramid bands
        if self.keep_gaussian:
            self.gbands = [None] * (self.height+1) # Gaussian pyramid bands

    def get_freqs(self):
        return self.band_freqs

    def get_band_count(self):
        return self.height+1

    def get_lband(self, band):
        if band == 0 or band == (len(self.lbands)-1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        return self.lbands[band] * band_mul

    def set_lband(self, band, data):
        if band == 0 or band == (len(self.lbands)-1):
            band_mul = 1.0
        else:
            band_mul = 2.0

        self.lbands[band] = data / band_mul

    def get_gband(self, band):
        return self.gbands[band]

    # def clear(self):
    #     for pyramid in self.P:
    #         for level in pyramid:
    #             # print ("deleting " + str(level))
    #             del level

    def decompose(self, image): 
        return self.laplacian_pyramid_dec(image, self.height+1)

    def reconstruct(self):
        img = self.lbands[-1]

        for i in reversed(range(0, len(self.lbands)-1)):
            img = self.gausspyr_expand(img, [self.lbands[i].shape[-2], self.lbands[i].shape[-1]])
            img += self.lbands[i]

        return img

    def laplacian_pyramid_dec(self, image, levels = -1, kernel_a = 0.4):
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return

        lpyr = []
        for i in range(height-1):
            layer = gpyr[i] - self.gausspyr_expand(gpyr[i+1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
            lpyr.append(layer)

        lpyr.append(gpyr[height-1])
        self.lbands = lpyr

        if self.keep_gaussian:
            self.gbands = gpyr        


# This pyramid computes and stores contrast during decomposition, improving performance and reducing memory consumption
class weber_contrast_pyr(lpyr_dec):

    def __init__(self, W, H, ppd, device, contrast):
        super().__init__(W, H, ppd, device)
        self.contrast = contrast

    def decompose(self, image):
        levels = self.height+1
        kernel_a = 0.4
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return []

        lpyr = []
        L_bkg_pyr = []
        for i in range(height):
            is_baseband = (i==(height-1))

            if is_baseband:
                layer = gpyr[i]
                if self.contrast.endswith('ref'):
                    L_bkg = torch.clamp(gpyr[i][...,1:2,:,:,:], min=0.01)
                else:
                    L_bkg = torch.clamp(gpyr[i][...,0:2,:,:,:], min=0.01)
                    # The sustained channels use the mean over the image as the background. Otherwise, they would be divided by itself and the contrast would be 1.
                    L_bkg_mean = torch.mean(L_bkg, dim=[-1, -2], keepdim=True)
                    L_bkg = L_bkg.repeat([int(image.shape[-4]/2), 1, 1, 1])
                    L_bkg[0:2,:,:,:] = L_bkg_mean
            else:
                glayer_ex = self.gausspyr_expand(gpyr[i+1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
                layer = gpyr[i] - glayer_ex 

                # Order: test-sustained-Y, ref-sustained-Y, test-rg, ref-rg, test-yv, ref-yv, test-transient-Y, ref-transient-Y
                # L_bkg is set to ref-sustained 
                if self.contrast == 'weber_g1_ref':
                    L_bkg = torch.clamp(glayer_ex[...,1:2,:,:,:], min=0.01)
                elif self.contrast == 'weber_g1':
                    L_bkg = torch.clamp(glayer_ex[...,0:2,:,:,:], min=0.01)
                elif self.contrast == 'weber_g0_ref':
                    L_bkg = torch.clamp(gpyr[i][...,1:2,:,:,:], min=0.01)
                else:
                    raise RuntimeError( f"Contrast {self.contrast} not supported")

            if L_bkg.shape[-4]==2: # If L_bkg NOT identical for the test and reference images
                contrast = torch.empty_like(layer)
                contrast[...,0::2,:,:,:] = torch.clamp(torch.div(layer[...,0::2,:,:,:], L_bkg[...,0,:,:,:]), max=1000.0)    
                contrast[...,1::2,:,:,:] = torch.clamp(torch.div(layer[...,1::2,:,:,:], L_bkg[...,1,:,:,:]), max=1000.0)    
            else:
                contrast = torch.clamp(torch.div(layer, L_bkg), max=1000.0)

            lpyr.append(contrast)
            L_bkg_pyr.append(torch.log10(L_bkg))

        # L_bkg_bb = gpyr[height-1][...,0:2,:,:,:]
        # lpyr.append(gpyr[height-1]) # Base band
        # L_bkg_pyr.append(L_bkg_bb) # Base band

        return lpyr, L_bkg_pyr


# This pyramid computes and stores contrast during decomposition, improving performance and reducing memory consumption
class log_contrast_pyr(lpyr_dec):

    def __init__(self, W, H, ppd, device, contrast):
        super().__init__(W, H, ppd, device)
        self.contrast = contrast

        # Assuming D65, there is a linear mapping from log10(L)+log10(M) to log10-luminance
        lms_d65 = [0.7347, 0.3163, 0.0208]
        self.a = 0.5
        self.b = math.log10(lms_d65[0]) - math.log10(lms_d65[1]) + math.log10(lms_d65[0]+lms_d65[1])

    def decompose(self, image):
        levels = self.height+1
        kernel_a = 0.4
        gpyr = self.gaussian_pyramid_dec(image, levels, kernel_a)

        height = len(gpyr)
        if height == 0:
            return []

        lpyr = []
        L_bkg_pyr = []
        for i in range(height):
            is_baseband = (i==(height-1))

            if is_baseband:
                contrast = gpyr[i]
                L_bkg = self.a * (gpyr[i][...,0:2,:,:,:] - self.b)
            else:
                glayer_ex = self.gausspyr_expand(gpyr[i+1], [gpyr[i].shape[-2], gpyr[i].shape[-1]], kernel_a)
                contrast = gpyr[i] - glayer_ex 

                # Order: test-sustained-Y, ref-sustained-Y, test-rg, ref-rg, test-yv, ref-yv, test-transient-Y, ref-transient-Y                
                # Mapping from log10(L) + log10(M) to log10(L+M)
                L_bkg = self.a * (glayer_ex[...,0:2,:,:,:] - self.b)

            lpyr.append(contrast)
            L_bkg_pyr.append(L_bkg)

        
        return lpyr, L_bkg_pyr
    
    
### Propagation    
def ITF_diff(field, z, dx, dy, wavelength, sideband=True, aperture=1.0):
    """
    Differentiable Incoherent Transfer Function
    """
    m, n = field.shape[-2::]
    Lx, Ly = (dx * n), (dy * m)
    device = field.device
    eps = 1e-6  # Small constant for numerical stability
    
    wavelength = torch.tensor([wavelength], dtype=torch.float32).to(device)
    z = torch.tensor([z], dtype=torch.float32).to(device)
    
    # Smooth versions of angular calculations
    angX = torch.asin(wavelength / (2 * dx))
    angY = torch.asin(wavelength / (2 * dy))
    
    # Smooth absolute value using sqrt(x^2 + ε)
    marX = torch.sqrt(z*z + eps) * torch.tan(angX)
    marY = torch.sqrt(z*z + eps) * torch.tan(angY)
    
    # Convert to integers while maintaining gradients using soft rounding
    marX_soft = torch.ceil(marX / dx)
    marY_soft = torch.ceil(marY / dy)
    marX_int = int(marX_soft.item())
    marY_int = int(marY_soft.item())
    
    pad_field = torch.nn.functional.pad(field, (marX_int, marX_int, marY_int, marY_int)).to(field.device)

    fy = torch.linspace(-1/(2*dy) + 0.5/(2*Ly), 1/(2*dy) - 0.5/(2*Ly), m + 2*marY_int).to(device)
    fx = torch.linspace(-1/(2*dx) + 0.5/(2*Lx), 1/(2*dx) - 0.5/(2*Lx), n + 2*marX_int).to(device)
    dfx = (1/dx) / n
    dfy = (1/dy) / m
    fY, fX = torch.meshgrid(fy, fx, indexing='ij')

    # Smooth aperture filter using sigmoid
    if sideband:
        aperture = (aperture/2, aperture)
    else:
        aperture = (aperture, aperture)

    # Smooth rectangular fourier filter using sigmoid
    temp = 50.0  # Temperature parameter for sigmoid sharpness
    nfX = fX / (torch.sqrt(torch.max(fX*fX) + eps))
    nfY = fY / (torch.sqrt(torch.max(fY*fY) + eps))
    
    BL_FILTER = torch.sigmoid(-temp*(torch.abs(nfY) - aperture[0])) * \
                torch.sigmoid(-temp*(torch.abs(nfX) - aperture[1]))
    
    # Energy normalization with stability
    BL_FILTER = BL_FILTER / torch.sqrt(torch.sum(BL_FILTER*BL_FILTER) + eps)

    # Smooth transfer function
    GammaSq = (1/wavelength)**2 - fX**2 - fY**2
    GammaSq = torch.maximum(GammaSq, torch.tensor(eps).to(device))  # Ensure positive
    TF = torch.exp(-2j * torch.pi * torch.sqrt(GammaSq) * z)
    TF = TF * BL_FILTER

    # FFT operations (these are differentiable)
    cpsf = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(TF), norm='ortho'))
    ipsf = torch.abs(cpsf) * torch.abs(cpsf)  # Smoother than **2
    OTF = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ipsf), norm='ortho'))

    # Smooth frequency cutoff
    max_fx = 1 / (wavelength * torch.sqrt((2 * dfx * z)**2 + 1))
    max_fy = 1 / (wavelength * torch.sqrt((2 * dfy * z)**2 + 1))
    
    FT = torch.sigmoid(-temp*(torch.abs(fX) - max_fx)) * \
        torch.sigmoid(-temp*(torch.abs(fY) - max_fy))

    AS = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(pad_field), norm='ortho'))
    PropagatedField = torch.sqrt((torch.fft.ifftshift(torch.fft.ifftn(
        torch.fft.ifftshift(AS * OTF * FT), norm='ortho'))).abs()**2 + eps)
    
    out = PropagatedField[marY_int:m+marY_int, marX_int:n+marX_int]
    return out.unsqueeze(0).unsqueeze(0)


def ITF(field, z, dx, dy, wavelength, sideband=True, aperture=1.0):
    """
    Simulate Incoherent Transfer Fuction
        :param field: shape of (H,W), tensor
        :param z: scalar, propagation distance
        :param dx, dy: pixel pitch
        :return out: shape of (1,1,H,W)
    """
    m, n = field.shape[-2::]
    Lx, Ly = (dx * n), (dy * m)

    # angX = math.asin(wavelength / (2 * dx))
    # angY = math.asin(wavelength / (2 * dy))
    # marX = math.fabs(z) * math.tan(angX)
    # marX = math.ceil(marX / dx)
    # marY = math.fabs(z) * math.tan(angY)
    # marY = math.ceil(marY / dy)
    device = field.device
    
    wavelength = torch.tensor([wavelength], dtype=torch.float32).to(device)
    z = torch.tensor([z], dtype=torch.float32).to(device)
    
    angX = torch.asin(wavelength / (2 * dx))
    angY = torch.asin(wavelength / (2 * dy))
    marX = torch.abs(z) * torch.tan(angX)
    marX = int(torch.ceil(marX / dx).item())
    marY = torch.abs(z) * torch.tan(angY)
    marY = int(torch.ceil(marY / dy).item())
    
    
    pad_field = torch.nn.functional.pad(field, (marX, marX, marY, marY)).to(field.device)

    fy = torch.linspace(-1 / (2 * dy) + 0.5 / (2 * Ly), 1 / (2 * dy) - 0.5 / (2 * Ly), m+ 2*marY).to(field.device)
    fx = torch.linspace(-1 / (2 * dx) + 0.5 / (2 * Lx), 1 / (2 * dx) - 0.5 / (2 * Lx), n+ 2*marX).to(field.device)
    dfx = (1 / dx) / n
    dfy = (1 / dy) / m
    fY, fX = torch.meshgrid(fy, fx, indexing='ij')

    # aperture for bandlimit
    if sideband:
        aperture = (aperture/2, aperture)
    else:
        aperture = (aperture, aperture)

    # rectangular fourier filter
    nfX = fX / torch.max(fX.abs())
    nfY = fY / torch.max(fY.abs())
    BL_FILTER = (nfY.abs() < aperture[0]) * (nfX.abs() < aperture[1])
    # energy normalization
    BL_FILTER = BL_FILTER / torch.sqrt(torch.sum(BL_FILTER) / torch.numel(BL_FILTER))
    # set transfer function
    GammaSq = (1 / wavelength) ** 2 - fX ** 2 - fY ** 2
    TF = torch.exp(-2 * 1j * math.pi * torch.sqrt(GammaSq) * z)
    TF = TF * BL_FILTER
    cpsf = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(TF), norm='ortho'))  # coherent psf
    ipsf = torch.abs(cpsf) ** 2  # incoherent psf
    OTF = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ipsf), norm='ortho'))

    max_fx = 1 / (wavelength * ((2 * dfx * z) ** 2 + 1) ** 0.5)
    max_fy = 1 / (wavelength * ((2 * dfy * z) ** 2 + 1) ** 0.5)
    FT = (torch.abs(fX) < max_fx) * (torch.abs(fY) < max_fy)  # Cutting aliasing
    AS = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(pad_field), norm='ortho'))
    PropagatedField = abs(torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(AS * OTF * FT), norm='ortho')))
    out = PropagatedField[marY:m+marY,marX:n+marX]  #imcrop

    return out.unsqueeze(0).unsqueeze(0)

def incoherent_focal_stack_rgbd(targets, masks, distances, dx, wavelengths, zero_padding = [True, False, True], apertures = 1., alpha = 0.5):
    """
    Generate incoherent focal stack using RGB-D images. 
    The occlusion mechanism is inspired from https://github.com/dongyeon93/holographic-parallax/blob/main/Incoherent_focal_stack.py
    
    Parameters
    ----------
    targets            : torch.tensor
                         Slices of the targets based on the depth masks.
    masks              : torch.tensor
                         Masks based on the depthmaps.
    distances          : list
                         A list of propagation distances.
    dx                 : float
                         Size of one single pixel in the field grid (in meters).
    wavelengths        : list
                         A list of wavelengths.
    zero_padding       : bool
                         Zero pad in Fourier domain.
    apertures           : torch.tensor
                         Fourier domain aperture (e.g., pinhole in a typical holographic display).
                         The default is one, but an aperture could be as large as input field [m x n].
    alpha              : float
                         Parameter to control how much the occlusion mask from the previous layer contributes to the current layer's occlusion when computing the focal stack.
    """
    distances = distances.flip(dims=(0,))    
    targets = targets.flip(dims=(0,))    
    masks = masks.flip(dims=(0,))    
    
    device = targets.device
    number_of_planes, number_of_channels, nu, nv = targets.shape
    focal_stack = torch.zeros_like(targets, dtype=torch.float32).to(device)
    
    aperture = [0.7053291536050157, 0.8653846153846154, 1.0]
    for ch, wavelength in enumerate(wavelengths):
        for n in range(number_of_planes):
            plane_sum = torch.zeros(nu, nv).to(device)
            occlusion_masks = torch.zeros(number_of_planes, nu, nv).to(device)
            
            for k in range(number_of_planes):
                distance = distances[n] - distances[k]
                mask_k = masks[k, ch]

                propagated_mask = ITF(mask_k, distance, dx, dx, wavelength, sideband=True, aperture=aperture[ch])
                
                # propagated_mask = torch.mean(propagated_mask, dim = 0)
                occlusion_mask = 1 - propagated_mask / (propagated_mask.max() if propagated_mask.max() else 1e-12)
                occlusion_masks[k, :, :] = torch.nan_to_num(occlusion_mask, 1.0)
                target = targets[k, ch]
                
                propagated_target = ITF(target, distance, dx, dx, wavelength, sideband=True, aperture=aperture[ch])
                if k == 0:
                    plane_sum = (1. * occlusion_mask) * plane_sum + propagated_target
                elif k == (number_of_planes - 1):
                    prev_occlusion_mask = occlusion_masks[k-1]
                    plane_sum = (alpha * occlusion_mask + (1.0 - alpha) * prev_occlusion_mask) * plane_sum + alpha * propagated_target
                else:
                    prev_occlusion_mask = occlusion_masks[k-1]
                    plane_sum = (alpha * occlusion_mask + (1.0 - alpha) * prev_occlusion_mask) * plane_sum + propagated_target
                
            focal_stack[n, ch, :, :] = plane_sum.abs()
            
    focal_stack = torch.flip(focal_stack, dims = (0,))
    return focal_stack / focal_stack.max()

def incoherent_focal_stack_rgbd_diff(targets, masks, distances, dx, wavelengths, zero_padding = [True, False, True], apertures = 1., alpha = 0.5):
    """
    Generate incoherent focal stack using RGB-D images. 
    The occlusion mechanism is inspired from https://github.com/dongyeon93/holographic-parallax/blob/main/Incoherent_focal_stack.py
    
    Parameters
    ----------
    targets            : torch.tensor
                         Slices of the targets based on the depth masks.
    masks              : torch.tensor
                         Masks based on the depthmaps.
    distances          : list
                         A list of propagation distances.
    dx                 : float
                         Size of one single pixel in the field grid (in meters).
    wavelengths        : list
                         A list of wavelengths.
    zero_padding       : bool
                         Zero pad in Fourier domain.
    apertures           : torch.tensor
                         Fourier domain aperture (e.g., pinhole in a typical holographic display).
                         The default is one, but an aperture could be as large as input field [m x n].
    alpha              : float
                         Parameter to control how much the occlusion mask from the previous layer contributes to the current layer's occlusion when computing the focal stack.
    """
    distances = distances.flip(dims=(0,))    
    targets = targets.flip(dims=(0,))    
    masks = masks.flip(dims=(0,))    
    
    device = targets.device
    number_of_planes, number_of_channels, nu, nv = targets.shape
    focal_stack = torch.zeros_like(targets, dtype=torch.float32).to(device)
    
    aperture = [0.7053291536050157, 0.8653846153846154, 1.0]
   
    for ch, wavelength in enumerate(wavelengths):
        for n in range(number_of_planes):
            plane_sum = torch.zeros(nu, nv).to(device)
            occlusion_masks = torch.zeros(number_of_planes, nu, nv).to(device)
            
            for k in range(number_of_planes):
                distance = distances[n] - distances[k]
                mask_k = masks[k]
                
                propagated_mask = ITF_diff(mask_k, distance, dx, dx, wavelength, sideband=True, aperture=aperture[ch])
                propagated_mask = propagated_mask.nan_to_num(0.0)
                
                occlusion_mask = 1 - propagated_mask / (propagated_mask.max() + 1e-6)
                occlusion_masks[k, :, :] = torch.nan_to_num(occlusion_mask, 1.0)
                target = targets[k, ch]
               

                propagated_target = ITF_diff(target, distance, dx, dx, wavelength, sideband=True, aperture=aperture[ch])
                if k == 0:
                    plane_sum = (1. * occlusion_mask) * plane_sum + propagated_target
                elif k == (number_of_planes - 1):
                    prev_occlusion_mask = occlusion_masks[k-1]
                    plane_sum = (alpha * occlusion_mask + (1.0 - alpha) * prev_occlusion_mask) * plane_sum + alpha * propagated_target
                else:
                    prev_occlusion_mask = occlusion_masks[k-1]
                    plane_sum = (alpha * occlusion_mask + (1.0 - alpha) * prev_occlusion_mask) * plane_sum + propagated_target
            focal_stack[n, ch, :, :] = plane_sum.abs()
            
    focal_stack = focal_stack.flip(dims=(0,))    
    return focal_stack / focal_stack.max()

def compute_focal_stack_distances(eyepiece, num_planes, prop_dists, depthmap_dists_range, device):
    """
    Calculate propagation and division dists for focal stack planes
        :param num_planes: scalar, number of fs planes
        :param depthmap_dists_range: scalar, total depth range of depthmap in mm. Reference plane is in the middle of this range
        :param eyepiece: scalar, focal length of eyepiece
        :param prop_dists: list of len 1, propagation distance of reference plane
        :return fs_dists: len num_planes list, propagation distance of fs planes in meters
        :return division_dists: len (num_planes-1) list, normalized to [0,1]. Depthmap is converted to mask using this
    """
    # dist to diopter
    far_near_dists = [eyepiece, eyepiece - depthmap_dists_range]
    far_near_diopters = [1/dist - 1/eyepiece for dist in far_near_dists]

    # linear spacing diopters
    division_diopters = torch.linspace(*far_near_diopters, 2 * num_planes - 1).to(device)
    prop_diopters = division_diopters[0::2]
    division_diopters = division_diopters[1::2]

    ref_plane_dist = depthmap_dists_range / 2
    # fs_dists = [prop_dists - ref_plane_dist + (eyepiece - 1 / (d + 1/eyepiece)) for d in prop_diopters]
    fs_dists = prop_dists - ref_plane_dist + (eyepiece - 1 / (prop_diopters + 1/eyepiece))
    # fs_dists = torch.flip(fs_dists, dims=(0,))
    return fs_dists

### Contrast
def get_contrast(image, saliency_map):
    device = image.device
    contrast_pyramid = []
    image = image.unsqueeze(0)
    lpyr = lpyr_dec_2(W=image.shape[-1], H=image.shape[-2], ppd=30, device=device, keep_gaussian=True)
    lpyr.decompose(image)

    rho_band = lpyr.get_freqs()
    rho_band[lpyr.get_band_count()-1] = 0.1 # Baseband
    sensitivity = contrast_sensitivity_function_Barten1999(rho_band)
    sensitivity /= torch.max(sensitivity)
    
    for b in range(lpyr.get_band_count()-1): # For each of the contrast band
        upsampled_gband = lpyr.gausspyr_expand(lpyr.get_gband(b+1), [lpyr.get_gband(b).shape[-2], lpyr.get_gband(b).shape[-1]], kernel_a=0.4)
        contrast = torch.abs(lpyr.get_lband(b) / upsampled_gband)
        contrast = torch.abs(contrast)
        interpolated_saliency = F.interpolate(saliency_map.unsqueeze(0).unsqueeze(0), contrast.shape[2:4]) 
        interpolated_saliency = max_min_normalization(interpolated_saliency)
        contrast = contrast * interpolated_saliency
        contrast = sensitivity[b] *  contrast 
        contrast_pyramid.append(contrast)
    return contrast_pyramid


### Helper functions

def find_local_maxima(tensor, window_size=3):
    """
    Find local maxima in the gradient of a 1D tensor using max pooling.
    
    Args:
    tensor (torch.Tensor): Input 1D tensor
    window_size (int): Size of the sliding window for max pooling
    
    Returns:
    torch.Tensor: 1D tensor containing the indices of local maxima in the gradient space
    """
    
    device = tensor.device
    if tensor.dim() != 1:
        raise ValueError("Input tensor must be 1-dimensional")
    
    gradient = torch.diff(tensor).to(device)
    
    gradient = gradient.unsqueeze(0).unsqueeze(0)
    
    padding = window_size // 2
    padded_gradient = F.pad(gradient, (padding, padding), mode='replicate')
    
    pooled = F.max_pool1d(padded_gradient, kernel_size=window_size, stride=1, padding=0).to(device)
    
    peaks = (gradient == pooled).squeeze() & (gradient.squeeze() > 0)
    
    peak_indices = torch.where(peaks)[0]
    
    return peak_indices + 1

def slice_rgbd_targets_diff(target, depth, depth_plane_positions, temperature=1e-6):
    """
    Differentiable and vectorized version of the RGBD target slicing function.
    
    Parameters
    ----------
    target                : torch.Tensor
                            The RGBD target tensor with shape (C, H, W).
    depth                 : torch.Tensor
                            The depth map corresponding to the target image with shape (H, W).
    depth_plane_positions : torch.Tensor
                            The positions of the depth planes used for slicing.
    temperature           : float
                            Temperature parameter for sharpness of transitions.
    
    Returns
    -------
    targets               : torch.Tensor
                            A tensor of shape (N, C, H, W) where N is the number of depth planes.
    masks                 : torch.Tensor
                            A tensor of shape (N, H, W) containing masks that sum to 1.
    """
    device = target.device
    number_of_planes = len(depth_plane_positions) - 1

    # Compute mid positions of depth planes
    mid_positions = (depth_plane_positions[:-1] + depth_plane_positions[1:]) / 2  # Shape: [N]

    # Expand dimensions for broadcasting
    depth_expanded = depth.unsqueeze(0)  # Shape: [1, H, W]
    mid_positions_expanded = mid_positions[:, None, None]  # Shape: [N, 1, 1]

    # Calculate distances to each plane position
    distances = torch.abs(depth_expanded - mid_positions_expanded)  # Shape: [N, H, W]

    # Convert distances to logits using negative exponential
    logits = -distances / temperature  # Shape: [N, H, W]

    # Apply softmax to get masks
    masks = F.softmax(logits, dim=0)  # Shape: [N, H, W]

    # Create targets
    targets = target.unsqueeze(0) * masks.unsqueeze(1)  # Shape: [N, C, H, W]

    return targets, masks