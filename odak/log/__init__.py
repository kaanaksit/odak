import os
import logging


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
    """
    formatter = logging.Formatter(
        fmt=logger_fmt,
        datefmt=logger_datefmt,
    )
    handler = logging.FileHandler(os.path.expanduser(logger_filename))
    handler.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    logger.addHandler(handler)
    logger.info("Logger initiated. Log is saved to {}.".format(logger_filename))
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
