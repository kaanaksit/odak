"""
``odak.wave``

Provides necessary definitions for merging geometric optics with wave theory and classical approaches
in the wave theory as well. See "Introduction to Fourier Optcs" from Joseph Goodman for the theoratical explanation.
"""

# To get sub-modules.
from .vector import *
from .utils import *
from .classical import *
from .lens import *
from ..tools import save_image
