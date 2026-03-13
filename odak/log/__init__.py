import os
import logging
import re


def _validate_path(path, allowed_extensions=None):
    """
    Inline path validation to avoid circular import issues.
    Validates for security safety similar to validate_path() in tools.file.
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
    if re.match(r"\\\\\.|\\\\?\.", path) or path.startswith("//."):
        raise ValueError("UNC/device paths not allowed")

    if len(resolved_path) > 260:  # Windows MAX_PATH limit
        raise ValueError("Path exceeds maximum allowed length (260 characters)")

    return resolved_path


def create_logger(
    logger_name,
    logger_filename,
    logger_fmt="%(asctime)s - %(message)s",
    logger_datefmt="%d-%b-%y %H:%M:%S",
    logger_level=logging.DEBUG,
):
    """
    Definiton to create a logger object using Python's built-in `logging` library.


    Parameters
    ----------
    logger_name     : string
                      Logger object name.
    logger_filename : string
                      Logger object's filename.
    logger_fmt      : string
                      Logger object's formatting.
    logger_datefmt      : string
                      Logger object's datetime formatting.
    logger_level    : int
                      Defaults is `logging.DEBUG`. For more details, refer to `logging` library's documentation.


    Returns
    ----------
    logger          : logging.Logger
                      For more details, refer to `logging` library's documentation.

    Raises
    ------
    ValueError      : If path validation fails or path traversal detected.
    TypeError       : If logger_filename is not a string.
    """
    formatter = logging.Formatter(
        fmt=logger_fmt,
        datefmt=logger_datefmt,
    )
    safe_path = _validate_path(logger_filename)
    handler = logging.FileHandler(safe_path)
    handler.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    logger.addHandler(handler)
    logger.info("Logger initiated. Log is saved to {}.".format(safe_path))
    return logger


filename_logger = "odak.log"
logger = create_logger(
    logger_name="odak",
    logger_filename=filename_logger,
    logger_fmt="%(asctime)s - %(message)s",
    logger_datefmt="%d-%b-%y %H:%M:%S",
    logger_level=logging.DEBUG,
)
logger.info(
    'Odak initiated a logger and logs are saved to "{}".'.format(filename_logger)
)
