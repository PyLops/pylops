"""
Medical Operators
=================

The subpackage medical provides linear operators and applications
aimed at solving various inverse problems in the area of Medical Imaging.

A list of operators present in pylops.medical:

    CT2D                            2D Computerized Tomography.

"""

from .ct import *

__all__ = [
    "CT2D",
]
