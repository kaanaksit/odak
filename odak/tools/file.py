import subprocess
import os
from odak import np
from PIL import Image

def save_image(fn,img,cmin=0,cmax=255):
    """
    Definition to save a Numpy array as an image.

    Parameters
    ----------
    fn           : str
                   Filename.
    img          : ndarray
                   A numpy array with NxMx3 or NxMx1 shapes.
    cmin         : int
                   Minimum value that will be interpreted as 0 level in the final image.
    cmax         : int
                   Maximum value that will be interpreted as 255 level in the final image.

    Returns
    ----------
    bool         :  bool
                    True if successful.

    """
    input_img                  = np.copy(img).astype(np.float)
    colorflag                  = False
    if  len(input_img.shape) == 3:
        if input_img.shape[2] > 1:
           input_img        = input_img[:,:,0:3]
           colorflag        = True
    input_img[input_img<cmin]  = cmin
    input_img[input_img>cmax]  = cmax
    input_img                 /= cmax
    input_img                 *= 255
    input_img                  = input_img.astype(np.uint8)
    if np.__name__ == 'cupy':
        input_img = np.asnumpy(input_img)
    if colorflag==True:
       result_img = Image.fromarray(input_img)
    elif colorflag == False:
       result_img = Image.fromarray(input_img).convert("L")
    result_img.save(fn)
    return True

def load_image(fn):
    """ 
    Definition to load an image from a given location as a Numpy array.

    Parameters
    ----------
    fn           : str
                   Filename.

    Returns
    ----------
    image        :  ndarray
                    Image loaded as a Numpy array.

    """
    image = Image.open(fn)
    return np.array(image)

def shell_command(cmd,cwd='.',timeout=0):
    """
    Definition to initiate shell commands.

    Parameters
    ----------
    cmd          : str
                   Command to be executed.
    cwd          : str
                   Working directory.
    timeout      : int
                   Timeout if the process isn't complete in the given number of seconds.


    Returns
    ----------
    bool         :  bool
                    True if succesful.

    """
    proc  = subprocess.Popen(
                             cmd,
                             cwd=cwd,
                             stdout=subprocess.PIPE
                            )
    try:
        outs, errs = proc.communicate(timeout=0)
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
    return True

def check_directory(directory):
    """
    Definition to check if a directory exist. If it doesn't exist, this definition will create one.

    Parameters
    ----------
    directory     : str
                    Full directory path.
    """
    if not os.path.exists(directory):
       os.makedirs(directory)
    return True
