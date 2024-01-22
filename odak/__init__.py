"""
``odak``
===================
Odak is the fundamental Python library needed for scientific computing in optical sciences. It includes modules for geometric raytracing and wave optics.

"""
import odak.tools
import odak.raytracing
import odak.wave
import odak.visualize
import odak.jones
import odak.catalog
import odak.measurement
import odak.learn
import odak.fit
import torch
import numpy as np
import logging


version_info = 0, 2, 6
__version__ = '.'.join(map(str, version_info))
pi = np.pi

filename_logger = 'odak.log'
logging.basicConfig(
                    filename = filename_logger, 
                    format = '%(asctime)s - %(message)s', 
                    datefmt = '%d-%b-%y %H:%M:%S',
                    level = logging.DEBUG
                   )
logging.debug('Odak\'s logger initiated. Log is saved to {}.'.format(filename_logger))
