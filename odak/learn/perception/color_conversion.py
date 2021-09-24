import torch

def rgb_2_ycrcb(image):
    ycrcb = torch.zeros(image.size()).to(image.device)
    ycrcb[:,0,:,:] = 0.299 * image[:,0,:,:] + 0.587 * image[:,1,:,:] + 0.114 * image[:,2,:,:]
    ycrcb[:,1,:,:] = 0.5 + 0.713 * (image[:,0,:,:] - ycrcb[:,0,:,:])
    ycrcb[:,2,:,:] = 0.5 + 0.564 * (image[:,2,:,:] - ycrcb[:,0,:,:])
    return ycrcb

def ycrcb_2_rgb(image):
    rgb = torch.zeros(image.size(), device=image.device)
    rgb[:,0,:,:] = image[:,0,:,:] + 1.403 * (image[:,1,:,:] - 0.5)
    rgb[:,1,:,:] = image[:,0,:,:] - 0.714 * (image[:,1,:,:] - 0.5) - 0.344 * (image[:,2,:,:] - 0.5)
    rgb[:,2,:,:] = image[:,0,:,:] + 1.773 * (image[:,2,:,:] - 0.5)
    return rgb
