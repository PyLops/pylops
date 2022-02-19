"""
AVO Operators
=============

The subpackage avo provides linear operators and applications aimed at
solving various inverse problems in the area of Seismic Reservoir
Characterization.

A list of available operators present in pylops.avo:

    AVOLinearModelling	                    AVO modelling.
    PoststackLinearModelling                Post-stack seismic modelling.
    PrestackLinearModelling                 Pre-stack seismic modelling.
    PrestackWaveletModelling                Pre-stack modelling operator for wavelet.

and a list of applications:

    PoststackInversion                      Post-stack seismic inversion.
    PrestackInversion                       Pre-stack seismic inversion.

"""

# isort: skip_file

from .poststack import *
from .prestack import *


__all__ = [
    "AVOLinearModelling",
    "PoststackLinearModelling",
    "PrestackWaveletModelling",
    "PrestackLinearModelling",
    "PoststackInversion",
    "PrestackInversion",
]
