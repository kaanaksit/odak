import torch
import odak.learn.perception.color_conversion as color_conversion

def color_map(input_image, target_image, model = 'Lab Stats'):
    """
    Internal function to map the color of an image to another image.
    Reference: Color transfer between images, Reinhard et al., 2001.
    
    Parameters
    ----------
    image               : torch.Tensor
                          Input image in RGB color space [3 x m x n].
    target_image        : torch.Tensor
    
    Returns
    -------
    mapped_image           : torch.Tensor
                             Input image with the color the distribution of the target image [3 x m x n].
    """
    if model == 'Lab Stats':
        lab_input = color_conversion.srgb_to_lab(input_image)
        lab_target = color_conversion.srgb_to_lab(target_image)
        input_mean_L = torch.mean(lab_input[0, :, :])
        input_mean_a = torch.mean(lab_input[1, :, :])
        input_mean_b = torch.mean(lab_input[2, :, :])
        input_std_L = torch.std(lab_input[0, :, :])
        input_std_a = torch.std(lab_input[1, :, :])
        input_std_b = torch.std(lab_input[2, :, :])
        target_mean_L = torch.mean(lab_target[0, :, :])
        target_mean_a = torch.mean(lab_target[1, :, :])
        target_mean_b = torch.mean(lab_target[2, :, :])
        target_std_L = torch.std(lab_target[0, :, :])
        target_std_a = torch.std(lab_target[1, :, :])
        target_std_b = torch.std(lab_target[2, :, :])
        lab_input[0, :, :] = (lab_input[0, :, :] - input_mean_L) * (target_std_L / input_std_L) + target_mean_L
        lab_input[1, :, :] = (lab_input[1, :, :] - input_mean_a) * (target_std_a / input_std_a) + target_mean_a
        lab_input[2, :, :] = (lab_input[2, :, :] - input_mean_b) * (target_std_b / input_std_b) + target_mean_b
        mapped_image = color_conversion.lab_to_srgb(lab_input.permute(1, 2, 0))
        return mapped_image
