__all__ = [
    "get_module",
    "get_module_name",
    "get_array_module",
    "get_convolve",
    "get_fftconvolve",
    "get_oaconvolve",
    "get_correlate",
    "get_add_at",
    "get_block_diag",
    "get_toeplitz",
    "get_csc_matrix",
    "get_sparse_eye",
    "get_lstsq",
    "get_complex_dtype",
    "get_real_dtype",
    "to_numpy",
    "to_cupy_conditional",
]

import os
from types import ModuleType
from typing import Callable

import numpy as np
import numpy.typing as npt
import scipy.fft as sp_fft
from scipy.linalg import block_diag, lstsq, toeplitz
from scipy.signal import convolve, correlate, fftconvolve, oaconvolve
from scipy.sparse import csc_matrix, eye

from pylops.utils import deps
from pylops.utils.typing import DTypeLike, NDArray

if deps.cupy_enabled:
    import cupy as cp
    import cupyx
    import cupyx.scipy.fft as cp_fft
    from cupyx.scipy.linalg import block_diag as cp_block_diag
    from cupyx.scipy.linalg import toeplitz as cp_toeplitz
    from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
    from cupyx.scipy.sparse import eye as cp_eye

if deps.cusignal_enabled:
    import cusignal

if deps.jax_enabled:
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import block_diag as jnp_block_diag


def get_module(backend: str = "numpy") -> ModuleType:
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
    elif backend == "jax":
        ncp = jnp
    else:
        raise ValueError("backend must be numpy, cupy, or jax")
    return ncp


def get_module_name(mod: ModuleType) -> str:
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
    elif mod == jnp:
        backend = "jax"
    else:
        raise ValueError("module must be numpy, cupy, or jax")
    return backend


def get_array_module(x: npt.ArrayLike) -> ModuleType:
    """Returns correct numerical module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    mod : :obj:`func`
        Module to be used to process array
        (:mod:`numpy`, :mod:`cupy`, or , :mod:`jax`)

    """
    if deps.cupy_enabled or deps.jax_enabled:
        if isinstance(x, jnp.ndarray):
            return jnp
        elif deps.cupy_enabled:
            return cp.get_array_module(x)
        else:
            return np
    else:
        return np


def get_convolve(x: npt.ArrayLike) -> Callable:
    """Returns correct convolve module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    f : :obj:`func`
        Function to be used to process array

    """
    if deps.cusignal_enabled or deps.jax_enabled:
        if isinstance(x, jnp.ndarray):
            return jax.scipy.signal.convolve
        elif deps.cusignal_enabled and cp.get_array_module(x) == cp:
            return cusignal.convolution.convolve
        else:
            return convolve
    else:
        return convolve


def get_fftconvolve(x: npt.ArrayLike) -> Callable:
    """Returns correct fftconvolve module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    f : :obj:`func`
        Function to be used to process array

    """
    if deps.cusignal_enabled or deps.jax_enabled:
        if isinstance(x, jnp.ndarray):
            return jax.scipy.signal.fftconvolve
        elif deps.cusignal_enabled and cp.get_array_module(x) == cp:
            return cusignal.convolution.fftconvolve
        else:
            return fftconvolve
    else:
        return fftconvolve


def get_oaconvolve(x: npt.ArrayLike) -> Callable:
    """Returns correct oaconvolve module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    f : :obj:`func`
        Function to be used to process array

    """
    if deps.cupy_enabled or deps.jax_enabled:
        if isinstance(x, jnp.ndarray):
            raise NotImplementedError(
                "oaconvolve not implemented in "
                "jax. Consider using a different"
                "option..."
            )
        elif deps.cupy_enabled and cp.get_array_module(x) == cp:
            raise NotImplementedError(
                "oaconvolve not implemented in "
                "cupy/cusignal. Consider using a different"
                "option..."
            )
        else:
            return oaconvolve
    else:
        return oaconvolve


def get_correlate(x: npt.ArrayLike) -> Callable:
    """Returns correct correlate module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    f : :obj:`func`
        Function to be used to process array

    """
    if deps.cusignal_enabled or deps.jax_enabled:
        if isinstance(x, jnp.ndarray):
            return jax.scipy.signal.correlate
        elif deps.cusignal_enabled and cp.get_array_module(x) == cp:
            return cusignal.convolution.correlate
        else:
            return correlate
    else:
        return correlate


def get_add_at(x: npt.ArrayLike) -> Callable:
    """Returns correct add.at module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    f : :obj:`func`
        Function to be used to process array

    """
    if not deps.cupy_enabled:
        return np.add.at

    if cp.get_array_module(x) == np:
        return np.add.at
    else:
        return cupyx.scatter_add


def get_block_diag(x: npt.ArrayLike) -> Callable:
    """Returns correct block_diag module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    f : :obj:`func`
        Function to be used to process array

    """
    if not deps.cupy_enabled and not deps.jax_enabled:
        return block_diag

    if deps.jax_enabled and isinstance(x, jnp.ndarray):
        return jax.scipy.linalg.block_diag
    elif cp.get_array_module(x) == np:
        return block_diag
    else:
        return cp_block_diag


def get_toeplitz(x: npt.ArrayLike) -> Callable:
    """Returns correct toeplitz module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    f : :obj:`func`
        Function to be used to process array

    """
    if not deps.cupy_enabled:
        return toeplitz

    if cp.get_array_module(x) == np:
        return toeplitz
    else:
        return cp_toeplitz


def get_csc_matrix(x: npt.ArrayLike) -> Callable:
    """Returns correct csc_matrix module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    f : :obj:`func`
        Function to be used to process array

    """
    if not deps.cupy_enabled:
        return csc_matrix

    if cp.get_array_module(x) == np:
        return csc_matrix
    else:
        return cp_csc_matrix


def get_sparse_eye(x: npt.ArrayLike) -> Callable:
    """Returns correct sparse eye module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        Array

    Returns
    -------
    f : :obj:`func`
        Function to be used to process array

    """
    if not deps.cupy_enabled:
        return eye

    if cp.get_array_module(x) == np:
        return eye
    else:
        return cp_eye


def get_lstsq(x: npt.ArrayLike) -> Callable:
    """Returns correct lstsq module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    f : :obj:`func`
        Function to be used to process array

    """
    if not deps.cupy_enabled:
        return lstsq

    if cp.get_array_module(x) == np:
        return lstsq
    else:
        return cp.linalg.lstsq


def get_sp_fft(x: npt.ArrayLike) -> Callable:
    """Returns correct scipy.fft module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    f : :obj:`func`
        Function to be used to process array

    """
    if not deps.cupy_enabled:
        return sp_fft

    if cp.get_array_module(x) == np:
        return sp_fft
    else:
        return cp_fft


def get_complex_dtype(dtype: DTypeLike) -> DTypeLike:
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


def get_real_dtype(dtype: DTypeLike) -> DTypeLike:
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


def to_numpy(x: NDArray) -> NDArray:
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


def to_cupy_conditional(x: npt.ArrayLike, y: npt.ArrayLike) -> NDArray:
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
