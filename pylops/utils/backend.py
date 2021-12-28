import numpy as np
from scipy.linalg import block_diag, lstsq, toeplitz
from scipy.signal import convolve, correlate, fftconvolve, oaconvolve
from scipy.sparse import csc_matrix, eye

from pylops.utils import deps

if deps.cupy_enabled:
    import cupy as cp
    import cupyx
    from cupyx.scipy.linalg import block_diag as cp_block_diag
    from cupyx.scipy.linalg import toeplitz as cp_toeplitz
    from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
    from cupyx.scipy.sparse import eye as cp_eye

if deps.cusignal_enabled:
    import cusignal

cu_message = "cupy package not installed. Use numpy arrays of " "install cupy."

cusignal_message = (
    "cusignal package not installed. Use numpy arrays of" "install cusignal."
)


def get_module(backend="numpy"):
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
    if backend == "numpy":
        ncp = np
    elif backend == "cupy":
        ncp = cp
    else:
        raise ValueError("backend must be numpy or cupy")
    return ncp


def get_module_name(mod):
    """Returns name of numerical module based on input numerical module

    Parameters
    ----------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    Returns
    -------
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy``). This
        parameter will be used to choose how to create the random vectors.

    """
    if mod == np:
        backend = "numpy"
    elif mod == cp:
        backend = "cupy"
    else:
        raise ValueError("module must be numpy or cupy")
    return backend


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


def get_convolve(x):
    """Returns correct convolve module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if not deps.cupy_enabled:
        return convolve

    if cp.get_array_module(x) == np:
        return convolve
    else:
        if deps.cusignal_enabled:
            return cusignal.convolution.convolve
        else:
            raise ModuleNotFoundError(cusignal_message)


def get_fftconvolve(x):
    """Returns correct fftconvolve module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if not deps.cupy_enabled:
        return fftconvolve

    if cp.get_array_module(x) == np:
        return fftconvolve
    else:
        if deps.cusignal_enabled:
            return cusignal.convolution.fftconvolve
        else:
            raise ModuleNotFoundError(cusignal_message)


def get_oaconvolve(x):
    """Returns correct oaconvolve module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if not deps.cupy_enabled:
        return oaconvolve

    if cp.get_array_module(x) == np:
        return oaconvolve
    else:
        raise NotImplementedError(
            "oaconvolve not implemented in "
            "cupy/cusignal. Consider using a different"
            "option..."
        )


def get_correlate(x):
    """Returns correct correlate module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if not deps.cupy_enabled:
        return correlate

    if cp.get_array_module(x) == np:
        return correlate
    else:
        if deps.cusignal_enabled:
            return cusignal.convolution.correlate
        else:
            raise ModuleNotFoundError(cusignal_message)


def get_add_at(x):
    """Returns correct add.at module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if not deps.cupy_enabled:
        return np.add.at

    if cp.get_array_module(x) == np:
        return np.add.at
    else:
        return cupyx.scatter_add


def get_block_diag(x):
    """Returns correct block_diag module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if not deps.cupy_enabled:
        return block_diag

    if cp.get_array_module(x) == np:
        return block_diag
    else:
        return cp_block_diag


def get_toeplitz(x):
    """Returns correct toeplitz module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if not deps.cupy_enabled:
        return toeplitz

    if cp.get_array_module(x) == np:
        return toeplitz
    else:
        return cp_toeplitz


def get_csc_matrix(x):
    """Returns correct csc_matrix module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if not deps.cupy_enabled:
        return csc_matrix

    if cp.get_array_module(x) == np:
        return csc_matrix
    else:
        return cp_csc_matrix


def get_sparse_eye(x):
    """Returns correct sparse eye module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if not deps.cupy_enabled:
        return eye

    if cp.get_array_module(x) == np:
        return eye
    else:
        return cp_eye


def get_lstsq(x):
    """Returns correct lstsq module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy`)

    """
    if not deps.cupy_enabled:
        return lstsq

    if cp.get_array_module(x) == np:
        return lstsq
    else:
        return cp.linalg.lstsq


def get_complex_dtype(dtype):
    """Returns a complex type in the precision of the input type.

    Parameters
    ----------
    dtype : :obj:`numpy.dtype`
        Input dtype.

    Returns
    -------
    complex_dtype : :obj:`numpy.dtype`
        Complex output type.
    """
    return (np.ones(1, dtype=dtype) + 1j * np.ones(1, dtype=dtype)).dtype


def get_real_dtype(dtype):
    """Returns a real type in the precision of the input type.

    Parameters
    ----------
    dtype : :obj:`numpy.dtype`
        Input dtype.

    Returns
    -------
    real_dtype : :obj:`numpy.dtype`
        Real output type.
    """
    return np.real(np.ones(1, dtype)).dtype


def to_numpy(x):
    """Convert x to numpy array

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        Array to evaluate

    Returns
    -------
    x : :obj:`cupy.ndarray`
        Converted array

    """
    if deps.cupy_enabled:
        if cp.get_array_module(x) == cp:
            x = cp.asnumpy(x)
    return x


def to_cupy_conditional(x, y):
    """Convert y to cupy array conditional to x being a cupy array

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        Array to evaluate
    y : :obj:`numpy.ndarray`
        Array to convert

    Returns
    -------
    y : :obj:`cupy.ndarray`
        Converted array

    """
    if deps.cupy_enabled:
        if cp.get_array_module(x) == cp and cp.get_array_module(y) == np:
            y = cp.asarray(y)
    return y
