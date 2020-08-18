import numpy as np
from pylops.utils import deps

if deps.cupy_enabled:
    import cupy as cp


def get_module(backend='numpy'):
    """Returns correct numerical module based on backend string

    Parameters
    ----------
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy``). This
        parameter will be used to choose how to create the random vectors.

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if backend == 'numpy':
        ncp = np
    elif backend == 'cupy':
        ncp = cp
    else:
        raise ValueError('backend must be numpy or cupy')
    return ncp

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
