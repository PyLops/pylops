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
        norm="ortho",
        real=False,
        ifftshift_before=False,
        fftshift_after=False,
        dtype="complex128",
    ):
        self.dims = _value_or_list_like_to_array(dims)
        _raise_on_wrong_dtype(self.dims, np.integer, "dims")

        self.ndim = len(self.dims)

        dirs = _value_or_list_like_to_array(dir)
        _raise_on_wrong_dtype(dirs, np.integer, "dir")
        self.dir = normalize_axis_index(dirs[0], self.ndim)

        if nfft is None:
            nfft = self.dims[self.dir]
        else:
            nffts = _value_or_list_like_to_array(nfft)
            _raise_on_wrong_dtype(nffts, np.integer, "nfft")
            nfft = nffts[0]
        self.nfft = nfft

        self.norm = norm
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


class _BaseFFTND(LinearOperator):
    """Base class for N-dimensional fast Fourier Transform"""

    def __init__(
        self,
        dims,
        dirs=None,
        nffts=None,
        sampling=1.0,
        real=False,
        ifftshift_before=False,
        fftshift_after=False,
        dtype="complex128",
    ):
        self.dims = _value_or_list_like_to_array(dims)
        _raise_on_wrong_dtype(self.dims, np.integer, "dims")

        self.ndim = len(self.dims)

        dirs = _value_or_list_like_to_array(dirs)
        _raise_on_wrong_dtype(dirs, np.integer, "dirs")
        self.dirs = np.array([normalize_axis_index(d, self.ndim) for d in dirs])
        self.ndirs = len(self.dirs)
        if self.ndirs != len(np.unique(self.dirs)):
            warnings.warn(
                "At least one direction is repeated. This may cause unexpected results."
            )

        nffts = _value_or_list_like_to_array(nffts, repeat=self.ndirs)
        if len(nffts[np.equal(nffts, None)]) > 0:  # Found None(s) in nffts
            nffts[np.equal(nffts, None)] = np.array(
                [self.dims[d] for d, n in zip(dirs, nffts) if n is None]
            )
            nffts = nffts.astype(self.dims.dtype)
        self.nffts = nffts
        _raise_on_wrong_dtype(self.nffts, np.integer, "nffts")

        sampling = _value_or_list_like_to_array(sampling, repeat=self.ndirs)
        if np.issubdtype(sampling.dtype, np.integer):  # Promote to float64 if integer
            sampling = sampling.astype(np.float64)
        self.sampling = sampling
        _raise_on_wrong_dtype(self.sampling, np.floating, "sampling")

        self.ifftshift_before = _value_or_list_like_to_array(
            ifftshift_before, repeat=self.ndirs
        )
        _raise_on_wrong_dtype(self.ifftshift_before, bool, "ifftshift_before")

        self.fftshift_after = _value_or_list_like_to_array(
            fftshift_after, repeat=self.ndirs
        )
        _raise_on_wrong_dtype(self.fftshift_after, bool, "fftshift_after")

        if (
            self.ndirs != len(self.nffts)
            or self.ndirs != len(self.sampling)
            or self.ndirs != len(self.ifftshift_before)
            or self.ndirs != len(self.fftshift_after)
        ):
            raise ValueError(
                (
                    "`dirs`, `nffts`, `sampling`, `ifftshift_before` and "
                    "`fftshift_after` must the have same number of elements. Received "
                    f"{self.ndirs}, {len(self.nffts)}, {len(self.sampling)}, "
                    f"{len(self.ifftshift_before)} and {len(self.fftshift_after)}, "
                    "respectively."
                )
            )
        self.real = real

        fs = [
            np.fft.fftshift(np.fft.fftfreq(n, d=s))
            if fftshift
            else np.fft.fftfreq(n, d=s)
            for n, s, fftshift in zip(self.nffts, self.sampling, self.fftshift_after)
        ]
        if self.real:
            fs[-1] = np.fft.rfftfreq(self.nffts[-1], d=self.sampling[-1])
            if self.fftshift_after[-1]:
                warnings.warn(
                    (
                        "Using real=True and fftshift_after on the last direction. "
                        "fftshift should only be applied on directions with negative "
                        "and positive frequencies. When using FFTND with real=True, "
                        "are all directions except the last. If you wish to proceed "
                        "applying fftshift on a frequency axis with only positive "
                        "frequencies, ignore this message."
                    )
                )
                fs[-1] = np.fft.fftshift(fs[-1])
        self.fs = tuple(fs)
        self.dims_fft = self.dims.copy()
        self.dims_fft[self.dirs] = self.nffts
        if self.real:
            self.dims_fft[self.dirs[-1]] = self.nffts[-1] // 2 + 1
        self.shape = (int(np.prod(self.dims_fft)), int(np.prod(self.dims)))
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
