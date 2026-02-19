"""
``odak.learn``
==============

Provides necessary definitions for neural networks and learning algorithms.
The definitions are based on torch framework.

This package contains modules for:
- Wave propagation simulations
- Learning tools and utilities
- Perception-based learning
- Ray tracing algorithms
- Neural network models
- Lensless imaging techniques

"""

import odak.learn.wave
import odak.learn.tools
import odak.learn.perception
import odak.learn.raytracing
import odak.learn.models
import odak.learn.lensless

# Make modules available at package level
__all__ = [
    'wave',
    'tools',
    'perception',
    'raytracing',
    'models',
    'lensless'
]
