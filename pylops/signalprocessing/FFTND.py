import numpy as np
from pylops import LinearOperator


class FFTND(LinearOperator):
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
    dtype : :obj:`str`, optional
        Type of elements in input array

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
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
    def __init__(self, dims, dirs=(0, 1, 2), nffts=(None, None, None),
                 sampling=(1., 1., 1.), dtype='complex128'):
        # checks
        if len(dims) < 3:
            raise ValueError('provide at least three dimensions')
        if len(dirs) < 3:
            raise ValueError('provide at least three directions along which '
                             'fft is applied')
        if len(nffts) != 3:
            raise ValueError('provide at least three fft dimensions')
        if len(sampling) != 3:
            raise ValueError('provide at least three sampling steps')

        if len(dirs) != len(nffts) \
                or len(dirs) != len(sampling) \
                or len(nffts) != len(sampling):
            raise ValueError('dirs, nffts, and sampling must '
                             'have same number of elements')
        self.ndims = len(dirs)
        self.dirs = dirs
        self.nffts = np.array([int(nffts[i]) if nffts[i] is not None
                               else dims[self.dirs[i]]
                               for i in range(self.ndims)])
        self.fs = [np.fft.fftfreq(nfft, d=samp)
                   for nfft, samp in zip(self.nffts, sampling)]

        self.dims = np.array(dims)
        self.dims_fft = self.dims.copy()
        for direction, nfft in zip(self.dirs, self.nffts):
            self.dims_fft[direction] = nfft
        self.shape = (int(np.prod(self.dims_fft)), int(np.prod(self.dims)))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = np.reshape(x, self.dims)
        y = np.sqrt(1./np.prod(self.nffts))*np.fft.fftn(x, s=self.nffts,
                                                        axes=self.dirs)
        return y.ravel()

    def _rmatvec(self, x):
        x = np.reshape(x, self.dims_fft)
        y = np.sqrt(np.prod(self.nffts)) * np.fft.ifftn(x, s=self.nffts,
                                                        axes=self.dirs)
        for direction in self.dirs:
            y = np.take(y, range(self.dims[direction]), axis=direction)
        return y.ravel()
