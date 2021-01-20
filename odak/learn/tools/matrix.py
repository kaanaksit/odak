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
        m = torch.nn.ZeroPad2d((hx, hx, hy, hy))
    elif method == 'left aligned':
        m = torch.nn.ZeroPad2d((0, hx*2, 0, hy*2))
    field_zero_padded = m(field)
    if type(size) != type(None):
        field_zero_padded = field_zero_padded[0:size[0],0:size[1]]
    return field_zero_padded
