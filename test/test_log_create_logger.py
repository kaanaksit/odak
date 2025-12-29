import sys
import odak
import logging


def test(
         logger_name = 'test',
         logger_filename = 'test.log',
         directory = 'test_output',
        ):
    odak.tools.check_directory(directory)
    logger_filename = '{}/{}'.format(
                                     directory,
                                     logger_filename,
                                    )
    logger = odak.log.create_logger(
                                    logger_name = logger_name,
                                    logger_filename = logger_filename,
                                    logger_fmt = '%(asctime)s - %(message)s',
                                    logger_datefmt = '%d-%b-%y %H:%M:%S',
                                    logger_level = logging.DEBUG,
                                   )
    logger.info('This is a test.')
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
