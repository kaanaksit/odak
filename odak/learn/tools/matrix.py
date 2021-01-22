from odak import np
import torch,torch.nn

def zero_pad(field,size=None,method='center'):
    """
    Definition to zero pad a MxN array to 2Mx2N array.

    Parameters
    ----------
    field             : ndarray
                        Input field MxN array.
    size              : list
                        Size to be zeropadded.
    method            : str
                        Zeropad either by placing the content to center or to the left.

    Returns
    ----------
    field_zero_padded : ndarray
                        Zeropadded version of the input field.
    """
    if type(size) == type(None):
        hx = int(torch.ceil(torch.tensor([field.shape[0]/2])))
        hy = int(torch.ceil(torch.tensor([field.shape[1]/2])))
    else:
        hx = int(torch.ceil(torch.tensor([(size[0]-field.shape[0])/2])))
        hy = int(torch.ceil(torch.tensor([(size[1]-field.shape[1])/2])))
    if method == 'center':
        m = torch.nn.ZeroPad2d((hy, hy, hx, hx))
    elif method == 'left aligned':
        m = torch.nn.ZeroPad2d((0, hy*2, 0, hx*2))
    field_zero_padded = m(field)
    if type(size) != type(None):
        field_zero_padded = field_zero_padded[0:size[0],0:size[1]]
    return field_zero_padded

def crop_center(field,size=None):
    """
    Definition to crop the center of a field with 2Mx2N size. The outcome is a MxN array.

    Parameters
    ----------
    field       : ndarray
                  Input field 2Mx2N array.

    Returns
    ----------
    cropped     : ndarray
                  Cropped version of the input field.
    """
    if type(size) == type(None):
        qx      = int(torch.ceil(torch.tensor(field.shape[0])/4))
        qy      = int(torch.ceil(torch.tensor(field.shape[1])/4))
        cropped = field[qx:3*qx,qy:3*qy]
    else:
        cx      = int(torch.ceil(torch.tensor(field.shape[0]/2)))
        cy      = int(torch.ceil(torch.tensor(field.shape[1]/2)))
        hx      = int(torch.ceil(torch.tensor(size[0]/2)))
        hy      = int(torch.ceil(torch.tensor(size[1]/2)))
        cropped = field[cx-hx:cx+hx,cy-hy:cy+hy]
    return cropped

