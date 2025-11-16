import sys
import odak


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
    logger = odak.tools.create_logger(
                                      logger_name = logger_name,
                                      logger_filename = logger_filename,
                                     )
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
