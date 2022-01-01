import numpy as np

from pylops.signalprocessing._BaseFFTs import _BaseFFTND


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
