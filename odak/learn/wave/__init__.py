"""
``odak.learn.wave``

Provides necessary definitions for neural networks and learning algorithms. The definitions are based on torch framework. Provides necessary definitions for merging geometric optics with wave theory and classical approaches in the wave theory as well. See "Introduction to Fourier Optcs" from Joseph Goodman for the theoratical explanation.

"""
import numpy as np
import torch
from .classical import *
from .legacy import *
from .lens import *
from .loss import * 
from .models import *
from .optimizers import * 
from .propagators import *
from .util import *
from .incoherent_propagation import *