import logging
import warnings

import numpy as np
from numpy.core.multiarray import normalize_axis_index

from pylops import LinearOperator
from pylops.utils.backend import get_complex_dtype, get_real_dtype

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def _value_or_list_like_to_array(value_or_list_like, repeat=1):
    """Convert an object which is either single value or a list-like to an array.
    Parameters
    ----------
    value_or_list_like
        Single value or list-like.
    repeat : `obj`:`int`
        Size of resulting array if value is passed. If list is passed, it is ignored.
    Returns
    -------
    out : `obj`:`numpy.array`
        When the input is a single value, returned an array with `repeat` samples
        containing that value. When the input is a list-like object, converts it to an
        array.
    """
    try:
        len(value_or_list_like)
        out = np.array(value_or_list_like)
    except TypeError:
        out = np.array([value_or_list_like] * repeat)
    return out


def _raise_on_wrong_dtype(arr, dtype, name):
    """Raises an error if dtype of `arr` is not a subdtype of `dtype`.
    Parameters
    ----------
    arr : `obj`:`numpy.array`
        Array whose type will be checked
    dtype : `obj`:`numpy.dtype`
        Type which must be a supertype of `arr.dtype`.
    name : `obj`:`str`
        Name of parameter to issue error.
    Raises
    ------
    TypeError
        When `arr.dtype` is not a subdtype of `dtype`.
    """
    if not np.issubdtype(arr.dtype, dtype):
        raise TypeError(
            (
                f"Wrong input type for `{name}`. "
                f'Must be "{dtype}", but received to "{arr.dtype}".'
            )
        )


class _BaseFFT(LinearOperator):
    """Base class for one dimensional Fast-Fourier Transform"""

    def __init__(
        self,
        dims,
        dir=0,
        nfft=None,
        sampling=1.0,
        real=False,
        ifftshift_before=False,
        fftshift_after=False,
        dtype="complex128",
    ):
        self.dims = _value_or_list_like_to_array(dims)
        _raise_on_wrong_dtype(self.dims, np.integer, "dims")

        self.ndim = len(self.dims)

        dirs = _value_or_list_like_to_array(dir)
        _raise_on_wrong_dtype(dirs, np.integer, "dirs")
        self.dir = normalize_axis_index(dirs[0], self.ndim)

        if nfft is None:
            nfft = self.dims[self.dir]
        else:
            nffts = _value_or_list_like_to_array(nfft)
            _raise_on_wrong_dtype(nffts, np.integer, "nfft")
            nfft = nffts[0]
        self.nfft = nfft

        self.real = real

        self.ifftshift_before = ifftshift_before

        self.f = (
            np.fft.rfftfreq(self.nfft, d=sampling)
            if real
            else np.fft.fftfreq(self.nfft, d=sampling)
        )
        self.fftshift_after = fftshift_after
        if self.fftshift_after:
            if self.real:
                warnings.warn(
                    "Using fftshift_after with real=True. fftshift should only be applied after a complex FFT. This is rarely intended behavior but if it is, ignore this message."
                )
            self.f = np.fft.fftshift(self.f)

        self.dims_fft = self.dims.copy()
        self.dims_fft[self.dir] = self.nfft // 2 + 1 if self.real else self.nfft
        self.shape = (int(np.prod(self.dims_fft)), int(np.prod(self.dims)))

        # Find types to enforce to forward and adjoint outputs. This is
        # required as np.fft.fft always returns complex128 even if input is
        # float32 or less. Moreover, when choosing real=True, the type of the
        # adjoint output is forced to be real even if the provided dtype
        # is complex.
        self.rdtype = get_real_dtype(dtype) if self.real else np.dtype(dtype)
        self.cdtype = get_complex_dtype(dtype)
        self.dtype = self.cdtype
        self.clinear = False if self.real or np.issubdtype(dtype, np.floating) else True
        self.explicit = False

    def _matvec(self, x):
        raise NotImplementedError(
            "_BaseFFT does not provide _matvec. It must be implemented separately."
        )

    def _rmatvec(self, x):
        raise NotImplementedError(
            "_BaseFFT does not provide _rmatvec. It must be implemented separately."
        )
