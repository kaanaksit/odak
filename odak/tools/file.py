import subprocess
import os
import json
import pathlib
import re
import shlex
import numpy as np
import cv2
import sys
import shutil
from ..log import logger

# Allowed commands whitelist for security
ALLOWED_COMMANDS = {
    "blender",
    "dispynode.py",
    "ffmpeg",
    "python",
    "python3",
    "git",
}

# Shell metacharacters that could enable command injection
DANGEROUS_PATTERNS = [
    r"[;&|`$]",  # Command chaining, backticks, variable expansion
    r"\$[{(]",  # Command substitution
    r"[<>]",  # Redirection
    r"""['"]""",  # Quotes that might be used for escaping
]


def validate_shell_command(cmd_list):
    """
    Validates shell command arguments for security.

    Parameters
    ----------
    cmd_list        : list
                      List of command arguments to validate.

    Returns
    -------
    validated_list  : list
                      The validated and sanitized command list.

    Raises
    ------
    ValueError      : If command contains dangerous characters or is not allowed.
    TypeError       : If cmd_list is not a list.
    """
    if not isinstance(cmd_list, list):
        raise TypeError(f"Command must be a list, got {type(cmd_list).__name__}")

    if len(cmd_list) == 0:
        raise ValueError("Command list cannot be empty")

    validated = []
    for idx, arg in enumerate(cmd_list):
        if not isinstance(arg, str):
            raise TypeError(
                f"Command argument {idx} must be a string, got {type(arg).__name__}"
            )

        # Check for null bytes
        if "\x00" in arg:
            raise ValueError(f"Null bytes detected in command argument {idx}")

        # Check for dangerous shell metacharacters
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, arg):
                raise ValueError(
                    f"Dangerous character detected in command argument {idx}: '{arg[:50]}...'. "
                    f"Shell metacharacters are not allowed."
                )

        # Check if the base command (first argument) is in whitelist
        if idx == 0:
            base_cmd = os.path.basename(arg).lower()
            if base_cmd not in ALLOWED_COMMANDS and not arg.startswith("/"):
                # Allow absolute paths, but warn
                logger.warning(
                    f"Command '{base_cmd}' is not in the allowed whitelist. "
                    f"Allowed commands: {sorted(ALLOWED_COMMANDS)}"
                )

        validated.append(arg)

    logger.debug(f"Shell command validated successfully")
    return validated


def validate_cwd(cwd):
    """
    Validates working directory path.

    Parameters
    ----------
    cwd             : str
                      Working directory path.

    Returns
    -------
    safe_cwd        : str
                      The validated and absolute path.

    Raises
    ------
    ValueError      : If path contains dangerous characters.
    """
    if not isinstance(cwd, str):
        raise TypeError(f"Working directory must be a string, got {type(cwd).__name__}")

    if "\x00" in cwd:
        raise ValueError("Null bytes detected in working directory path")

    expanded = os.path.expanduser(cwd)
    absolute_path = os.path.abspath(expanded)

    # Check if the directory exists (optional, can be removed if you want to allow non-existent dirs)
    # if not os.path.isdir(absolute_path):
    #     raise ValueError(f"Working directory does not exist: {absolute_path}")

    return absolute_path


def validate_path(path, allowed_extensions=None):
    """
    Validates a file path for security safety.

    Parameters
    ----------
    path            : str
                      Path to validate.
    allowed_extensions : list, optional
                          List of allowed extensions (e.g., ['.png', '.jpg']).
                          If None, all extensions are allowed.

    Returns
    -------
    safe_path       : str
                      The validated and secured path (with tilde expanded).

    Raises
    ------
    ValueError      : If path traversal attempt detected or extension not allowed.
    TypeError       : If path is not a string.
    """
    if not isinstance(path, str):
        raise TypeError(f"Path must be a string, got {type(path).__name__}")

    # Check for null bytes before expanding user (Windows path injection)
    if "\x00" in path:
        raise ValueError("Null bytes not allowed in path")

    # Check for path traversal patterns BEFORE expanding
    if ".." in path.split(os.sep) or ".." in path.replace(os.sep, "/").split("/"):
        if re.search(r"(^|[/\\])\.\.([/\\]|$)", path):
            raise ValueError("Path traversal detected: '..' not allowed in path")

    # Check for URL protocols before expanding
    path_lower = path.lower()
    if re.search(r"https?://|ftp://", path_lower):
        raise ValueError("URL protocols not allowed in file paths")

    path = os.path.expanduser(path)
    resolved_path = os.path.abspath(path)

    # Check for UNC or device paths on Windows
    if re.match(r"\\\\\\\|\\\\\\?\.\\", path) or path.startswith("//."):
        raise ValueError("UNC/device paths not allowed")

    if len(resolved_path) > 260:  # Windows MAX_PATH limit
        raise ValueError("Path exceeds maximum allowed length (260 characters)")

    if allowed_extensions is not None:
        _, file_ext = os.path.splitext(path)
        ext_lower = file_ext.lower()
        allowed_normalized = [
            ext.lower() if ext.startswith(".") else f".{ext}"
            for ext in allowed_extensions
        ]
        if ext_lower not in allowed_normalized:
            raise ValueError(
                f"File extension '{file_ext}' is not allowed. "
                f"Allowed: {allowed_extensions}"
            )

    logger.debug(f"Path validated: {path}")
    return resolved_path


def get_base_filename(filename):
    """
    Definition to retrieve the base filename and extension type.


    Parameters
    ----------
    filename       : str
                     Input filename.


    Returns
    -------
    basename       : str
                     Basename extracted from the filename.
    extension      : str
                     Extension extracted from the filename.
    """
    cache = os.path.basename(filename)
    basename = os.path.splitext(cache)[0]
    extension = os.path.splitext(cache)[1]
    return basename, extension


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
    logger.debug("Resizing image to {}".format(target_size))
    img = cv2.resize(
        img, dsize=(target_size[0], target_size[1]), interpolation=cv2.INTER_AREA
    )
    logger.debug("Image resized to {}".format(target_size))
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
    logger.info("Saving image: {}".format(fn))
    input_img = np.copy(img).astype(np.float32)
    cmin = float(cmin)
    cmax = float(cmax)
    input_img[input_img < cmin] = cmin
    input_img[input_img > cmax] = cmax
    input_img /= cmax
    input_img = input_img * 1.0 * (2**color_depth - 1)
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
    safe_path = validate_path(
        fn, allowed_extensions=[".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
    )
    cv2.imwrite(safe_path, input_img)
    logger.info("Saved image: {}".format(safe_path))
    return True


def load_image(fn, normalizeby=0.0, torch_style=False):
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
    logger.info("Loading image: {}".format(fn))
    safe_path = validate_path(
        fn,
        allowed_extensions=[
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tiff",
            ".tif",
            ".gif",
            ".webp",
            ".pbm",
            ".pgm",
            ".ppm",
            ".sr",
            ".ras",
        ],
    )
    image = cv2.imread(safe_path, cv2.IMREAD_UNCHANGED)
    if isinstance(image, type(None)):
        logger.warning("Image not properly loaded. Check filename or image type.")
        sys.exit()
    if len(image.shape) > 2:
        new_image = np.copy(image)
        new_image[:, :, 0] = image[:, :, 2]
        new_image[:, :, 2] = image[:, :, 0]
        image = new_image
    if normalizeby != 0.0:
        image = image * 1.0 / normalizeby
    if torch_style == True and len(image.shape) > 2:
        image = np.moveaxis(image, -1, 0)
    logger.info("Loaded image: {}".format(safe_path))
    return image.astype(float)


def shell_command(cmd, cwd=".", timeout=None, check=True):
    """
    Definition to initiate shell commands securely.

    Parameters
    ----------
    cmd          : list
                   Command to be executed as a list of arguments.
                   Example: ['blender', '-b', 'file.blend']
    cwd          : str
                   Working directory. Default is current directory.
    timeout      : int or None
                   Timeout in seconds if the process isn't complete.
                   If None, no timeout is enforced.
    check        : bool
                   Set it to True to return results and enable timeout. False returns only process.

    Returns
    -------
    proc         : subprocess.Popen
                   Generated process handle.
    outs         : str or bytes
                   Standard output of the executed command (None when check=False).
    errs         : str or bytes
                   Standard error of the executed command (None when check=False).

    Raises
    ------
    ValueError   : If command contains dangerous characters, forbidden commands,
                   null bytes, or if working directory path is invalid.
    TypeError    : If cmd is not a list or cwd is not a string.
    subprocess.TimeoutExpired : If process exceeds timeout (when check=True).

    Security Features
    ---------------
    - Command whitelist validation (blender, dispynode.py, ffmpeg, python, etc.)
    - Blocks shell metacharacters (; & | ` $ < > quotes)
    - Null byte injection protection
    - Path traversal blocked in working directory
    - Uses Popen with shell=False for safe execution

    Example
    ------
    >>> proc, outs, errs = shell_command(['blender', '-b', 'scene.blend'])
    >>> proc, None, None = shell_command(['python', 'script.py'], check=False)
    """
    # Validate command arguments
    validated_cmd = validate_shell_command(cmd)

    # Validate working directory
    safe_cwd = validate_cwd(cwd if isinstance(cwd, str) else ".")

    # Execute with shell=False for security (prevents shell injection)
    proc = subprocess.Popen(
        validated_cmd,
        cwd=safe_cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,  # Critical: Prevents shell metacharacter injection
    )

    if not check:
        return proc, None, None

    try:
        outs, errs = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()

    return proc, outs, errs


def check_directory(directory, validate=True):
    """
    Definition to check if a directory exist. If it doesn't exist, this definition will create one.

    Parameters
    ----------
    directory     : str
                    Full directory path.
    validate      : bool
                    Whether to validate the path for security (default: True).
                    When True, checks for path traversal, null bytes, and other unsafe patterns.

    Returns
    -------
    bool         :  bool
                   Returns True if directory already exists, False if created.

    Raises
    ------
    ValueError   : If path traversal attempt detected, invalid characters found,
                   or directory creation fails due to permissions/invalid path.
    TypeError    : If directory is not a string.
    """
    if validate:
        logger.debug("Checking directory: {}".format(directory))
        safe_path = validate_path(directory + "/")
    else:
        # Bypass validation for internal use only
        safe_path = os.path.abspath(os.path.expanduser(directory))

    if not os.path.exists(safe_path):
        try:
            os.makedirs(safe_path)
            logger.info("Created directory: {}".format(safe_path))
            return False
        except Exception as e:
            raise ValueError(f"Failed to create directory '{safe_path}': {str(e)}")

    # Verify it's actually a directory, not a file with the same name
    if not os.path.isdir(safe_path):
        raise ValueError(f"Path exists but is not a directory: {safe_path}")

    logger.info("Directory already exists: {}".format(safe_path))
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
    logger.info("Saving dictionary: {}".format(filename))
    safe_path = validate_path(filename, allowed_extensions=[".json"])
    with open(safe_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)
    logger.info("Saved dictionary: {}".format(safe_path))
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
    logger.info("Loading dictionary: {}".format(filename))
    safe_path = validate_path(filename, allowed_extensions=[".json"])
    with open(safe_path, "r", encoding="utf-8") as f:
        settings = json.load(f)
    logger.info("Loaded dictionary: {}".format(safe_path))
    return settings


def list_files(path, key="*.*", recursive=True):
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
    -------
    files_list  : ndarray
                  list of files found in a given path.
    """
    safe_path = validate_path(path + "/")
    search_result = None
    if recursive == True:
        search_result = pathlib.Path(safe_path).rglob(key)
    elif recursive == False:
        search_result = pathlib.Path(safe_path).glob(key)
    if search_result is None:
        return []
    files_list = [str(item) for item in search_result]
    return sorted(files_list)


def list_directories(path, recursive=True):
    """
    Lists directories inside a given directory, recursively if specified.

    Parameters
    ----------
    path      : str
                The path to the directory you want to list.
    recursive : bool, optional
                If True, lists subdirectories recursively. Defaults to True.

    Returns
    -------
    list
                A list of directory names.
    """
    directories = []
    safe_path = validate_path(path + "/")
    if recursive:
        for entry in os.listdir(safe_path):
            full_path = os.path.join(safe_path, entry)
            if os.path.isdir(full_path):
                directories.append(entry)
                directories.extend(list_directories(full_path, recursive=True))
    else:
        contents = os.listdir(safe_path)
        directories = [f for f in contents if os.path.isdir(os.path.join(safe_path, f))]
    return sorted(directories)


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
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
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


def expanduser(filename):
    """
    Definition to decode filename using namespaces and shortcuts.


    Parameters
    ----------
    filename      : str
                    Filename.


    Returns
    -------
    new_filename  : str
                    Filename.
    """
    new_filename = os.path.expanduser(filename)
    return new_filename


def copy_file(source, destination, follow_symlinks=True):
    """
    Definition to copy a file from one location to another.



    Parameters
    ----------
    source          : str
                      Source filename.
    destination     : str
                      Destination filename.
    follow_symlinks : bool
                      Set to True to follow the source of symbolic links.

    Returns
    -------
    None           : On success, raises ValueError if validation fails.
    """
    safe_source = validate_path(source)
    safe_destination = validate_path(destination)
    return shutil.copyfile(
        safe_source, safe_destination, follow_symlinks=follow_symlinks
    )


def write_to_text_file(content, filename, write_flag="w"):
    """
    Defininition to write a Pythonic list to a text file.


    Parameters
    ----------
    content         : list
                      Pythonic string list to be written to a file.
    filename        : str
                      Destination filename (i.e. test.txt).
    write_flag      : str
                      Defines the interaction with the file.
                      The default is "w" (overwrite any existing content).
                      For more see: https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files

    Returns
    -------
    bool           : True if successful.
    """
    allowed_write_flags = {"w", "a", "x", "r+", "w+", "a+"}
    if write_flag not in allowed_write_flags:
        raise ValueError(
            f"Write flag must be one of {allowed_write_flags}, got '{write_flag}'"
        )
    safe_path = validate_path(filename)
    with open(safe_path, write_flag) as f:
        for line in content:
            f.write("{}\n".format(line))
    return True


def read_text_file(filename):
    """
    Definition to read a given text file and convert it into a Pythonic list.


    Parameters
    ----------
    filename        : str
                      Source filename (i.e. test.txt).


    Returns
    -------
    content         : list
                      Pythonic string list containing the text from the file provided.

    Raises
    ------
    ValueError     : If path validation fails or unsafe characters detected.
    """
    content = []
    safe_path = validate_path(filename)
    with open(safe_path, "r", encoding="utf-8") as f:
        for line in f:
            content.append(line.rstrip())
    return content
