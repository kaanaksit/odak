import torch


def check_loss_inputs(loss_name, image, target):
        if image.size() != target.size():
            raise Exception(
                loss_name + " ERROR: Input and target must have same dimensions.")
        if len(image.size()) != 4:
            raise Exception(
                loss_name + " ERROR: Input and target must have 4 dimensions in total (NCHW format).")
        if image.size(1) != 1 and image.size(1) != 3:
            raise Exception(
                loss_name + """ ERROR: Inputs should have either 1 or 3 channels 
                (1 channel for grayscale, 3 for RGB or YCrCb).
                Ensure inputs have 1 or 3 channels and are in NCHW format.""")

def slice_rgbd_targets(target, depth, depth_plane_positions):
    """
    Slices the target RGBD image and depth map into multiple layers based on depth plane positions.
    
    Parameters
    ----------
    target                 : torch.Tensor
                             The RGBD target tensor with shape (C, H, W).
    depth                  : torch.Tensor
                             The depth map corresponding to the target image with shape (H, W).
    depth_plane_positions  : list or torch.Tensor
                             The positions of the depth planes used for slicing.
    
    Returns
    -------
    targets              : torch.Tensor
                           A tensor of shape (N, C, H, W) where N is the number of depth planes. Contains the sliced targets for each depth plane.
    masks                : torch.Tensor
                           A tensor of shape (N, C, H, W) containing binary masks for each depth plane.
    """
    device = target.device
    number_of_planes = len(depth_plane_positions) - 1
    targets = torch.zeros(
                        number_of_planes,
                        target.shape[0],
                        target.shape[1],
                        target.shape[2],
                        requires_grad = False,
                        device = device
                        )
    masks = torch.zeros_like(targets, dtype = torch.int).to(device)
    mask_zeros = torch.zeros_like(depth, dtype = torch.int)
    mask_ones = torch.ones_like(depth, dtype = torch.int)
    for i in range(1, number_of_planes+1):
        for ch in range(target.shape[0]):
            pos = depth_plane_positions[i] 
            prev_pos = depth_plane_positions[i-1] 
            if i <= (number_of_planes - 1):
                condition = torch.logical_and(prev_pos <= depth, depth < pos)
            else:
                condition = torch.logical_and(prev_pos <= depth, depth <= pos)
            mask = torch.where(condition, mask_ones, mask_zeros)
            new_target = target[ch] * mask
            targets[i-1, ch] = new_target.squeeze(0)
            masks[i-1, ch] = mask.detach().clone() 
    return targets, masks