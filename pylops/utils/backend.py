__all__ = [
    "get_module",
    "get_module_name",
    "get_array_module",
    "get_normalize_axis_index",
    "get_convolve",
    "get_fftconvolve",
    "get_oaconvolve",
    "get_correlate",
    "get_add_at",
    "get_sliding_window_view",
    "get_block_diag",
    "get_toeplitz",
    "get_csc_matrix",
    "get_sparse_eye",
    "get_lstsq",
    "get_sp_fft",
    "get_complex_dtype",
    "get_real_dtype",
    "to_cupy",
    "to_numpy",
    "to_cupy_conditional",
    "to_numpy_conditional",
    "inplace_set",
    "inplace_add",
    "inplace_multiply",
    "inplace_divide",
    "randn",
]

from types import ModuleType
from typing import Callable

import numpy as np
import scipy.fft as sp_fft
from scipy.linalg import block_diag, lstsq, toeplitz
from scipy.signal import convolve, correlate, fftconvolve, oaconvolve
from scipy.sparse import csc_matrix, eye

from pylops.utils import deps
from pylops.utils.typing import ArrayLike, DTypeLike, NDArray, Tfftengine_ncj

if deps.cupy_enabled:
    import cupy as cp
    import cupyx
    import cupyx.scipy.fft as cp_fft
    from cupyx.scipy.linalg import block_diag as cp_block_diag
    from cupyx.scipy.linalg import toeplitz as cp_toeplitz
    from cupyx.scipy.signal import convolve as cp_convolve
    from cupyx.scipy.signal import correlate as cp_correlate
    from cupyx.scipy.signal import fftconvolve as cp_fftconvolve
    from cupyx.scipy.signal import oaconvolve as cp_oaconvolve
    from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
    from cupyx.scipy.sparse import eye as cp_eye

if deps.jax_enabled:
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import block_diag as jnp_block_diag
    from jax.scipy.linalg import toeplitz as jnp_toeplitz
    from jax.scipy.signal import convolve as j_convolve
    from jax.scipy.signal import fftconvolve as j_fftconvolve

# need to check numpy version since the namespace of normalize_axis_index
# changed from numpy>=2.0.0
np_version = np.__version__.split(".")
if int(np_version[0]) > 1:
    from numpy.lib.array_utils import normalize_axis_index
else:
    from numpy.core.multiarray import normalize_axis_index


def get_module(backend: Tfftengine_ncj = "numpy") -> ModuleType:
    """Returns correct numerical module based on backend string

    Parameters
    ----------
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy`` or ``jax``). This
        parameter will be used to choose how to create the random vectors.

    Returns
    -------
    mod : :obj:`callable`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy` or :mod:`jax`)

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
    mod : :obj:`callable`
        Module to be used to process array (:mod:`numpy` or :mod:`cupy` or :mod:`jax`)

    Returns
    -------
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy`` or ``jax``). This
        parameter will be used to choose how to create the random vectors.

    """
    if mod == np:
        backend = "numpy"
    elif deps.cupy_enabled and mod == cp:
        backend = "cupy"
    elif deps.jax_enabled and mod == jnp:
        backend = "jax"
    else:
        raise ValueError("module must be numpy, cupy, or jax")
    return backend


def get_array_module(x: ArrayLike) -> ModuleType:
    """Returns correct numerical module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray` or :obj:`jax.Array`
        Array

    Returns
    -------
    mod : :obj:`callable`
        Module to be used to process array
        (:mod:`numpy`, :mod:`cupy`, or , :mod:`jax`)

    """
    if deps.cupy_enabled or deps.jax_enabled:
        if deps.jax_enabled and isinstance(x, jnp.ndarray):
            return jnp
        elif deps.cupy_enabled:
            return cp.get_array_module(x)
        else:
            return np
    else:
        return np


def get_normalize_axis_index() -> Callable:
    """Returns correct normalize_axis_index module based on numpy version

    Returns
    -------
    f : :obj:`callable`
        Function to be used to process array

    """
    return normalize_axis_index


def get_convolve(x: ArrayLike) -> Callable:
    """Returns correct convolve module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray` or :obj:`jax.Array`
        Array

    Returns
    -------
    f : :obj:`callable`
        Function to be used to process array

    """
    if deps.cupy_enabled or deps.jax_enabled:
        if deps.jax_enabled and isinstance(x, jnp.ndarray):
            return j_convolve
        elif deps.cupy_enabled and cp.get_array_module(x) == cp:
            return cp_convolve
        else:
            return convolve
    else:
        return convolve


def get_fftconvolve(x: ArrayLike) -> Callable:
    """Returns correct fftconvolve module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray` or :obj:`jax.Array`
        Array

    Returns
    -------
    f : :obj:`callable`
        Function to be used to process array

    """
    if deps.cupy_enabled or deps.jax_enabled:
        if deps.jax_enabled and isinstance(x, jnp.ndarray):
            return j_fftconvolve
        elif deps.cupy_enabled and cp.get_array_module(x) == cp:
            return cp_fftconvolve
        else:
            return fftconvolve
    else:
        return fftconvolve


def get_oaconvolve(x: ArrayLike) -> Callable:
    """Returns correct oaconvolve module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray` or :obj:`jax.Array`
        Array

    Returns
    -------
    f : :obj:`callable`
        Function to be used to process array

    """
    if deps.cupy_enabled or deps.jax_enabled:
        if deps.jax_enabled and isinstance(x, jnp.ndarray):
            raise NotImplementedError(
                "oaconvolve not implemented in "
                "jax. Consider using a different"
                "option..."
            )
        elif deps.cupy_enabled and cp.get_array_module(x) == cp:
            return cp_oaconvolve
        else:
            return oaconvolve
    else:
        return oaconvolve


def get_correlate(x: ArrayLike) -> Callable:
    """Returns correct correlate module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray` or :obj:`jax.Array`
        Array

    Returns
    -------
    f : :obj:`callable`
        Function to be used to process array

    """
    if deps.cupy_enabled or deps.jax_enabled:
        if deps.jax_enabled and isinstance(x, jnp.ndarray):
            return jax.scipy.signal.correlate
        elif deps.cupy_enabled and cp.get_array_module(x) == cp:
            return cp_correlate
        else:
            return correlate
    else:
        return correlate


def get_add_at(x: ArrayLike) -> Callable:
    """Returns correct add.at module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray` or :obj:`jax.Array`
        Array

    Returns
    -------
    f : :obj:`callable`
        Function to be used to process array

    """
    if not deps.cupy_enabled:
        return np.add.at

    if cp.get_array_module(x) == np:
        return np.add.at
    else:
        return cupyx.scatter_add


def get_sliding_window_view(x: ArrayLike) -> Callable:
    """Returns correct sliding_window_view module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray` or :obj:`jax.Array`
        Array

    Returns
    -------
    f : :obj:`callable`
        Function to be used to process array

    """
    if not deps.cupy_enabled:
        return np.lib.stride_tricks.sliding_window_view

    if cp.get_array_module(x) == np:
        return np.lib.stride_tricks.sliding_window_view
    else:
        return cp.lib.stride_tricks.sliding_window_view


def get_block_diag(x: ArrayLike) -> Callable:
    """Returns correct block_diag module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray` or :obj:`jax.Array`
        Array

    Returns
    -------
    f : :obj:`callable`
        Function to be used to process array

    """
    if deps.cupy_enabled or deps.jax_enabled:
        if deps.jax_enabled and isinstance(x, jnp.ndarray):
            return jnp_block_diag
        elif deps.cupy_enabled and cp.get_array_module(x) == cp:
            return cp_block_diag
        else:
            return block_diag
    else:
        return block_diag


def get_toeplitz(x: ArrayLike) -> Callable:
    """Returns correct toeplitz module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    f : :obj:`callable`
        Function to be used to process array

    """
    if deps.cupy_enabled or deps.jax_enabled:
        if deps.jax_enabled and isinstance(x, jnp.ndarray):
            return jnp_toeplitz
        elif deps.cupy_enabled and cp.get_array_module(x) == cp:
            return cp_toeplitz
        else:
            return toeplitz
    else:
        return toeplitz


def get_csc_matrix(x: ArrayLike) -> Callable:
    """Returns correct csc_matrix module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    f : :obj:`callable`
        Function to be used to process array

    """
    if not deps.cupy_enabled:
        return csc_matrix

    if cp.get_array_module(x) == np:
        return csc_matrix
    else:
        return cp_csc_matrix


def get_sparse_eye(x: ArrayLike) -> Callable:
    """Returns correct sparse eye module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        Array

    Returns
    -------
    f : :obj:`callable`
        Function to be used to process array

    """
    if not deps.cupy_enabled:
        return eye

    if cp.get_array_module(x) == np:
        return eye
    else:
        return cp_eye


def get_lstsq(x: ArrayLike) -> Callable:
    """Returns correct lstsq module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    f : :obj:`callable`
        Function to be used to process array

    """
    if not deps.cupy_enabled:
        return lstsq

    if cp.get_array_module(x) == np:
        return lstsq
    else:
        return cp.linalg.lstsq


def get_sp_fft(x: ArrayLike) -> Callable:
    """Returns correct scipy.fft module based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Array

    Returns
    -------
    f : :obj:`callable`
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


def to_cupy(x: ArrayLike) -> ArrayLike:
    """Convert x to cupy array if cupy is available

    Parameters
    ----------
    x : :obj:`numpy.ndarray`, :obj:`cupy.ndarray` or :obj:`jax.Array`
        Array to evaluate

    Returns
    -------
    x : :obj:`numpy.ndarray`, :obj:`cupy.ndarray` or :obj:`jax.Array`
        Converted array

    """
    if deps.cupy_enabled:
        if cp.get_array_module(x) != cp:
            x = cp.asarray(x)
    return x


def to_numpy(x: ArrayLike) -> NDArray:
    """Convert x to numpy array

    Parameters
    ----------
    x : :obj:`numpy.ndarray`, :obj:`cupy.ndarray` or :obj:`jax.Array`
        Array to evaluate

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Converted array

    """
    if deps.cupy_enabled:
        if cp.get_array_module(x) == cp:
            x = cp.asnumpy(x)
    if deps.jax_enabled:
        if isinstance(x, jnp.ndarray):
            x = np.array(x)
    return x


def to_cupy_conditional(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """Convert y to cupy array conditional to x being a cupy array

    Parameters
    ----------
    x : :obj:`numpy.ndarray`, :obj:`cupy.ndarray` or `jax.Array`
        Array to evaluate
    y : :obj:`numpy.ndarray`, :obj:`cupy.ndarray` or `jax.Array`
        Array to convert

    Returns
    -------
    y : :obj:`numpy.ndarray`, :obj:`cupy.ndarray` or `jax.Array`
        Converted array

    """
    if deps.cupy_enabled:
        if cp.get_array_module(x) == cp and cp.get_array_module(y) != cp:
            with cp.cuda.Device(x.device):
                y = cp.asarray(y)
    return y


def to_numpy_conditional(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """Convert y to numpy array conditional to x being a numpy array

    Parameters
    ----------
    x : :obj:`numpy.ndarray`, :obj:`cupy.ndarray` or `jax.Array`
        Array to evaluate
    y : :obj:`numpy.ndarray`, :obj:`cupy.ndarray` or `jax.Array`
        Array to convert

    Returns
    -------
    y : :obj:`numpy.ndarray`, :obj:`cupy.ndarray` or `jax.Array`
        Converted array

    """
    if deps.cupy_enabled:
        if cp.get_array_module(x) == np and cp.get_array_module(y) == cp:
            y = cp.asnumpy(y)
    if deps.jax_enabled:
        if isinstance(x, np.ndarray) and isinstance(y, jnp.ndarray):
            y = np.array(y)
    return y


def inplace_set(x: ArrayLike, y: ArrayLike, idx: list) -> NDArray:
    """Perform inplace set based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`jax.Array`
        Array whose values are placed at indices ``idx``
    y : :obj:`numpy.ndarray` or :obj:`jax.Array`
        Output array
    idx : :obj:`list`
        Indices where values ``x`` are set

    Returns
    -------
    y : :obj:`numpy.ndarray` or :obj:`jax.Array`
        Output array

    """
    if deps.jax_enabled and isinstance(x, jnp.ndarray):
        y = y.at[idx].set(x)
        return y
    else:
        y[idx] = x
        return y


def inplace_add(x: ArrayLike, y: ArrayLike, idx: list) -> NDArray:
    """Perform inplace add based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`jax.Array`
        Array to sum
    y : :obj:`numpy.ndarray` or :obj:`jax.Array`
        Output array
    idx : :obj:`list`
        Indices to sum at

    Returns
    -------
    y : :obj:`numpy.ndarray` or :obj:`jax.Array`
        Output array

    """
    if deps.jax_enabled and isinstance(x, jnp.ndarray):
        y = y.at[idx].add(x)
        return y
    else:
        y[idx] += x
        return y


def inplace_multiply(x: ArrayLike, y: ArrayLike, idx: list) -> NDArray:
    """Perform inplace multiplication based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`jax.Array`
        Array to sum
    y : :obj:`numpy.ndarray` or :obj:`jax.Array`
        Output array
    idx : :obj:`list`
        Indices to multiply at

    Returns
    -------
    y : :obj:`numpy.ndarray` or :obj:`jax.Array`
        Output array

    """
    if deps.jax_enabled and isinstance(x, jnp.ndarray):
        y = y.at[idx].multiply(x)
        return y
    else:
        y[idx] *= x
        return y


def inplace_divide(x: ArrayLike, y: ArrayLike, idx: list) -> NDArray:
    """Perform inplace division based on input

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`jax.Array`
        Array to sum
    y : :obj:`numpy.ndarray` or :obj:`jax.Array`
        Output array
    idx : :obj:`list`
        Indices to divide at

    Returns
    -------
    y : :obj:`numpy.ndarray` or :obj:`jax.Array`
        Output array

    """
    if deps.jax_enabled and isinstance(x, jnp.ndarray):
        y = y.at[idx].divide(x)
        return y
    else:
        y[idx] /= x
        return y


def randn(*n: int, backend: Tfftengine_ncj = "numpy") -> NDArray:
    """Returns randomly generated number

    Parameters
    ----------
    *n : :obj:`int`
        Number of samples to generate in each dimension
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy``). This
        parameter will be used to choose how to create the random vectors.

    Returns
    -------
    x : :obj:`numpy.ndarray` or :obj:`jax.Array`
        Generated array

    """
    if backend == "numpy":
        x = np.random.randn(*n)
    elif backend == "cupy":
        x = cp.random.randn(*n)
    elif backend == "jax":
        x = jnp.array(np.random.randn(*n))
    else:
        raise ValueError("backend must be numpy, cupy, or jax")
    return x
