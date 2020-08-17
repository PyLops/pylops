import numpy as np
from pylops.utils import deps

if deps.cupy_enabled:
    import cupy as cp


def get_array_module(x):
    """Returns correct numerical module based on input


    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if deps.cupy_enabled:
        return cp.get_array_module(x)
    else:
        return np