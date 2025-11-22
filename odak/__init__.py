"""
``odak``
===================
Odak is the fundamental Python library needed for scientific computing in light-related sciences including but not limited to Optics, Display & Camera Technology, Optical Computing, Perceptual Graphics, and Computer Graphics.
"""

import odak.log
import odak.tools
import odak.raytracing
import odak.wave
import odak.visualize
import odak.visualize.plotly
import odak.jones
import odak.catalog
import odak.measurement
import odak.learn
import odak.fit
import torch
import numpy as np

version_info = 0, 2, 7
__version__ = '.'.join(map(str, version_info))

pi = np.pi

