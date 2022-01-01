import logging
import warnings

import numpy as np
from numpy.core.multiarray import normalize_axis_index

from pylops import LinearOperator
from pylops.utils.backend import get_complex_dtype, get_real_dtype

from ._BaseFFTs import _raise_on_wrong_dtype, _value_or_list_like_to_array

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


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


class FFTND(_BaseFFTND):
    r"""N-dimensional Fast-Fourier Transform.
    Apply n-dimensional Fast-Fourier Transform (FFT) to any n axes
    of a multi-dimensional array depending on the choice of ``dirs``.
    Note that the FFTND operator is a simple overload to the numpy
    :py:func:`numpy.fft.fftn` in forward mode and to the numpy
    :py:func:`numpy.fft.ifftn` in adjoint mode, however scaling is taken
    into account differently to guarantee that the operator is passing the
    dot-test.
    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    dirs : :obj:`tuple`, optional
        Directions along which FFTND is applied
    nffts : :obj:`tuple`, optional
        Number of samples in Fourier Transform for each direction (same as
        input if ``nffts=(None, None, None, ..., None)``)
    sampling : :obj:`tuple`, optional
        Sampling steps in each direction
    real : :obj:`bool`, optional
        Model to which fft is applied has real numbers (``True``) or not
        (``False``). Used to enforce that the output of adjoint of a real
        model is real. Note that the real FFT is applied only to the first
        dimension to which the FFTND operator is applied (last element of
        `dirs`)
    ifftshift_before : :obj:`bool`, optional
        Apply ifftshift (``True``) or not (``False``) to model vector (before FFT).
        Consider using this option when the model vector's respective axis is symmetric
        with respect to the zero value sample. This will shift the zero value sample to
        coincide with the zero index sample. With such an arrangement, FFT will not
        introduce a sample-dependent phase-shift when compared to the continuous Fourier
        Transform.
        Defaults to not applying ifftshift.
    fftshift_after : :obj:`bool`, optional
        Apply fftshift (``True``) or not (``False``) to data vector (after FFT).
        Consider using this option when you require frequencies to be arranged
        naturally, from negative to positive. When not applying fftshift after FFT,
        frequencies are arranged from zero to largest positive, and then from negative
        Nyquist to the frequency bin before zero.
        Defaults to not applying fftshift.
    dtype : :obj:`str`, optional
        Type of elements in input array. Note that the `dtype` of the operator
        is the corresponding complex type even when a real type is provided.
        Nevertheless, the provided dtype will be enforced on the vector
        returned by the `rmatvec` method.
    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    clinear : :obj:`bool`
        Operator is complex-linear. Is false when either real=True or when
        dtype is not a complex type.
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (True) or not (False)
    Raises
    ------
    ValueError
        If ``dims``, ``dirs``, ``nffts``, or ``sampling`` have less than \
        three elements and if the dimension of ``dirs``, ``nffts``, and
        ``sampling`` is not the same
    Notes
    -----
    The FFTND operator applies the n-dimensional forward Fourier transform
    to a multi-dimensional array. Without loss of generality we consider here
    a three-dimensional signal :math:`d(z, y, x)`.
    The FFTND in forward mode is:
    .. math::
        D(k_z, k_y, k_x) = \mathscr{F} (d) = \int \int d(z,y,x) e^{-j2\pi k_zz}
        e^{-j2\pi k_yy} e^{-j2\pi k_xx} dz dy dx
    Similarly, the  three-dimensional inverse Fourier transform is applied to
    the Fourier spectrum :math:`D(k_z, k_y, k_x)` in adjoint mode:
    .. math::
        d(z, y, x) = \mathscr{F}^{-1} (D) = \int \int D(k_z, k_y, k_x)
        e^{j2\pi k_zz} e^{j2\pi k_yy} e^{j2\pi k_xx} dk_z dk_y  dk_x
    Both operators are effectively discretized and solved by a fast iterative
    algorithm known as Fast Fourier Transform.
    """

    def __init__(
        self,
        dims,
        dirs=(0, 1, 2),
        nffts=None,
        sampling=1.0,
        real=False,
        ifftshift_before=False,
        fftshift_after=False,
        dtype="complex128",
    ):
        super().__init__(
            dims=dims,
            dirs=dirs,
            nffts=nffts,
            sampling=sampling,
            real=real,
            ifftshift_before=ifftshift_before,
            fftshift_after=fftshift_after,
            dtype=dtype,
        )

    def _matvec(self, x):
        x = np.reshape(x, self.dims)
        if self.ifftshift_before.any():
            x = np.fft.ifftshift(x, axes=self.dirs[self.ifftshift_before])
        if not self.clinear:
            x = np.real(x)
        if self.real:
            y = np.fft.rfftn(x, s=self.nffts, axes=self.dirs, norm="ortho")
            # Apply scaling to obtain a correct adjoint for this operator
            y = np.swapaxes(y, -1, self.dirs[-1])
            y[..., 1 : 1 + (self.nffts[-1] - 1) // 2] *= np.sqrt(2)
            y = np.swapaxes(y, self.dirs[-1], -1)
        else:
            y = np.fft.fftn(x, s=self.nffts, axes=self.dirs, norm="ortho")
        y = y.astype(self.cdtype)
        if self.fftshift_after.any():
            y = np.fft.fftshift(y, axes=self.dirs[self.fftshift_after])
        return y.ravel()

    def _rmatvec(self, x):
        x = np.reshape(x, self.dims_fft)
        if self.fftshift_after.any():
            x = np.fft.ifftshift(x, axes=self.dirs[self.fftshift_after])
        if self.real:
            # Apply scaling to obtain a correct adjoint for this operator
            x = x.copy()
            x = np.swapaxes(x, -1, self.dirs[-1])
            x[..., 1 : 1 + (self.nffts[-1] - 1) // 2] /= np.sqrt(2)
            x = np.swapaxes(x, self.dirs[-1], -1)
            y = np.fft.irfftn(x, s=self.nffts, axes=self.dirs, norm="ortho")
        else:
            y = np.fft.ifftn(x, s=self.nffts, axes=self.dirs, norm="ortho")
        for direction in self.dirs:
            y = np.take(y, range(self.dims[direction]), axis=direction)
        if not self.clinear:
            y = np.real(y)
        y = y.astype(self.rdtype)
        if self.ifftshift_before.any():
            y = np.fft.fftshift(y, axes=self.dirs[self.ifftshift_before])
        return y.ravel()
