import subprocess
import os
import json
import pathlib
import numpy as np
import cv2


def resize_image(img, target_size):
    """
    Definition to resize a given image to a target shape.


    Parameters
    ----------
    img           : ndarray
                    MxN image to be resized.
                    Image must be normalized (0-1).
    target_size   : list
                    Target shape.


    Returns
    ----------
    img           : ndarray
                    Resized image.

    """
    img = cv2.resize(img, dsize=(target_size[0], target_size[1]), interpolation=cv2.INTER_AREA)
    return img


def save_image(fn, img, cmin=0, cmax=255, color_depth=8):
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
    color_depth  : int
                   Pixel color depth in bits, default is eight bits.

    Returns
    ----------
    bool         :  bool
                    True if successful.

    """
    input_img = np.copy(img).astype(np.float32)
    cmin = float(cmin)
    cmax = float(cmax)
    input_img[input_img < cmin] = cmin
    input_img[input_img > cmax] = cmax
    input_img /= cmax
    input_img = input_img * 1. * (2**color_depth - 1)
    if color_depth == 8:
        input_img = input_img.astype(np.uint8)
    elif color_depth == 16:
        input_img = input_img.astype(np.uint16)
    if len(input_img.shape) > 2:
        if input_img.shape[2] > 1:
            cache_img = np.copy(input_img)
            cache_img[:, :, 0] = input_img[:, :, 2]
            cache_img[:, :, 2] = input_img[:, :, 0]
            input_img = cache_img
    cv2.imwrite(fn, input_img)
    return True


def load_image(fn, normalizeby=0., torch_style=False):
    """ 
    Definition to load an image from a given location as a Numpy array.

    Parameters
    ----------
    fn           : str
                   Filename.
    normalizeby  : float
                   Value to to normalize images with. Default value of zero will lead to no normalization.
    torch_style  : bool
                   If set True, it will load an image mxnx3 as 3xmxn.

    Returns
    ----------
    image        :  ndarray
                    Image loaded as a Numpy array.

    """
    image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    if len(image.shape) > 2:
        new_image = np.copy(image)
        new_image[:, :, 0] = image[:, :, 2]
        new_image[:, :, 2] = image[:, :, 0]
        image = new_image
    if normalizeby != 0.:
        image = image * 1. / normalizeby
    if torch_style == True and len(image.shape) > 2:
        image = np.moveaxis(image, -1, 0)
    return image.astype(float)


def shell_command(cmd, cwd='.', timeout=None, check=True):
    """
    Definition to initiate shell commands.

    Parameters
    ----------
    cmd          : list
                   Command to be executed. 
    cwd          : str
                   Working directory.
    timeout      : int
                   Timeout if the process isn't complete in the given number of seconds.
    check        : bool
                   Set it to True to return the results and to enable timeout.

    Returns
    ----------
    proc         : subprocess.Popen
                   Generated process.
    outs         : str
                   Outputs of the executed command, returns None when check is set to False.
    errs         : str
                   Errors of the executed command, returns None when check is set to False.

    """
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE
    )
    if check == False:
        return proc, None, None
    try:
        outs, errs = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
    return proc, outs, errs


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
        return False
    return True


def save_dictionary(settings, filename):
    """
    Definition to load a dictionary (JSON) file.


    Parameters
    ----------
    settings      : dict
                    Dictionary read from the file.
    filename      : str
                    Filename.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)
    return settings


def load_dictionary(filename):
    """
    Definition to load a dictionary (JSON) file.


    Parameters
    ----------
    filename      : str
                    Filename.

    Returns
    ----------
    settings      : dict
                    Dictionary read from the file.

    """
    settings = json.load(open(filename))
    return settings


def list_files(path, key='*.*', recursive=True):
    """
    Definition to list files in a given path with a given key.

    Parameters
    ----------
    path        : str
                  Path to a folder.
    key         : str
                  Key used for scanning a path.
    recursive   : bool
                  If set True, scan the path recursively.

    Returns
    ----------
    files_list  : ndarray
                  list of files found in a given path.
    """
    if recursive == True:
        search_result = pathlib.Path(path).rglob(key)
    elif recursive == False:
        search_result = pathlib.Path(path).glob(key)
    files_list = []
    for item in search_result:
        files_list.append(str(item))
    files_list = sorted(files_list)
    return files_list


def convert_bytes(num):
    """
    A definition to convert bytes to semantic scheme (MB,GB or alike). Inspired from https://stackoverflow.com/questions/2104080/how-can-i-check-file-size-in-python#2104083.

    Parameters
    ----------
    num        : float
                 Size in bytes

    Returns
    ----------
    num        : float
                 Size in new unit.
    x          : str
                 New unit bytes, KB, MB, GB or TB.
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return num, x
        num /= 1024.0
    return None, None


def size_of_a_file(file_path):
    """
    A definition to get size of a file with a relevant unit.

    Parameters
    ----------
    file_path  : float
                 Path of the file.

    Returns
    ----------
    a          : float
                 Size of the file.
    b          : str
                 Unit of the size (bytes, KB, MB, GB or TB).
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        a, b = convert_bytes(file_info.st_size)
        return a, b
    return None, None
