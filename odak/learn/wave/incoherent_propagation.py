import torch
from odak.learn.wave import calculate_amplitude, wavenumber, propagate_beam
from odak.learn.perception.color_conversion import rgb_to_linear_rgb, linear_rgb_to_rgb


def incoherent_focal_stack_rgbd(targets, masks, distances, dx, wavelengths, zero_padding = [True, False, True], apertures = [1., 1., 1.], alpha = 0.5):
    """
    Generate incoherent focal stack using RGB-D images. 
    Please note that the targets and masks should follow the order from the furthest to the closest.
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
                         Fourier domain apertures (e.g., pinhole in a typical holographic display) for each color channel.
    alpha              : float
                         Parameter to control how much the occlusion mask from the previous layer contributes to the current layer's occlusion when computing the focal stack.
    """
    
    
    device = targets.device
    number_of_planes, number_of_channels, nu, nv = targets.shape
    focal_stack = torch.zeros_like(targets, dtype=torch.float32).to(device)
    for ch, wavelength in enumerate(wavelengths):
        for n in range(number_of_planes):
            plane_sum = torch.zeros(nu, nv).to(device)
            occlusion_masks = torch.zeros(number_of_planes, nu, nv).to(device)
            
            for k in range(number_of_planes):
                distance = distances[n] - distances[k]
                mask_k = masks[k]
                propagated_mask = propagate_beam(
                                                 field = mask_k,
                                                 k = wavenumber(wavelength),
                                                 distance = distance,
                                                 dx = dx,
                                                 wavelength = wavelength,
                                                 propagation_type = 'Incoherent Angular Spectrum',
                                                 zero_padding = zero_padding,
                                                 aperture = apertures[ch]
                                                )
                propagated_mask = calculate_amplitude(propagated_mask)
                propagated_mask = torch.mean(propagated_mask, dim = 0)
                occlusion_mask =  1.0 - propagated_mask / (propagated_mask.max() if propagated_mask.max() else 1e-12)
                occlusion_masks[k, :, :] = torch.nan_to_num(occlusion_mask, 1.0)
                target = targets[k, ch]
                propagated_target = propagate_beam(
                                                   field = target,
                                                   k = wavenumber(wavelength),
                                                   distance = distance,
                                                   dx = dx,
                                                   wavelength = wavelength,
                                                   propagation_type = 'Incoherent Angular Spectrum',
                                                   zero_padding = zero_padding,
                                                   aperture = apertures[ch]
                                                  )
                propagated_target = calculate_amplitude(propagated_target)
                if k == 0:
                    plane_sum = (1. * occlusion_mask) * plane_sum + propagated_target
                elif k == (number_of_planes - 1):
                    prev_occlusion_mask = occlusion_masks[k-1]
                    plane_sum = (alpha * occlusion_mask + (1.0 - alpha) * prev_occlusion_mask) * plane_sum + alpha * propagated_target
                else:
                    prev_occlusion_mask = occlusion_masks[k-1]
                    plane_sum = (alpha * occlusion_mask + (1.0 - alpha) * prev_occlusion_mask) * plane_sum + propagated_target
                
            focal_stack[n, ch, :, :] = plane_sum
    return focal_stack / focal_stack.max()