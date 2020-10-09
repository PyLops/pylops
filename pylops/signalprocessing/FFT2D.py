import numpy as np
from pylops import LinearOperator


class FFT2D(LinearOperator):
    r"""Two dimensional Fast-Fourier Transform.

    Apply two dimensional Fast-Fourier Transform (FFT) to any pair of axes of a
    multi-dimensional array depending on the choice of ``dirs``.
    Note that the FFT2D operator is a simple overload to the numpy
    :py:func:`numpy.fft.fft2` in forward mode and to the numpy
    :py:func:`numpy.fft.ifft2` in adjoint mode, however scaling is taken
    into account differently to guarantee that the operator is passing the
    dot-test.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    dirs : :obj:`tuple`, optional
        Pair of directions along which FFT2D is applied
    nffts : :obj:`tuple`, optional
        Number of samples in Fourier Transform for each direction (same as
        input if ``nffts=(None, None)``)
    sampling : :obj:`tuple`, optional
        Sampling steps ``dy`` and ``dx``
    dtype : :obj:`str`, optional
        Type of elements in input array. Note that the `dtype` of the operator
        is the corresponding complex type even when a real type is provided.
        Nevertheless, the provided dtype will be enforced on the vector
        returned by the `rmatvec` method.

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
        If ``dims`` has less than two elements, and if ``dirs``, ``nffts``,
        or ``sampling`` has more or less than two elements.

    Notes
    -----
    The FFT2D operator applies the two-dimensional forward Fourier transform
    to a signal :math:`d(y,x)` in forward mode:

    .. math::
        D(k_y, k_x) = \mathscr{F} (d) = \int \int d(y,x) e^{-j2\pi k_yy}
        e^{-j2\pi k_xx} dy dx

    Similarly, the  two-dimensional inverse Fourier transform is applied to
    the Fourier spectrum :math:`D(k_y, k_x)` in adjoint mode:

    .. math::
        d(y,x) = \mathscr{F}^{-1} (D) = \int \int D(k_y, k_x) e^{j2\pi k_yy}
        e^{j2\pi k_xx} dk_y  dk_x

    Both operators are effectively discretized and solved by a fast iterative
    algorithm known as Fast Fourier Transform.

    """
    def __init__(self, dims, dirs=(0, 1), nffts=(None, None),
                 sampling=(1., 1.), dtype='complex128'):
        # checks
        if len(dims) < 2:
            raise ValueError('provide at least two dimensions')
        if len(dirs) != 2:
            raise ValueError('provide at two directions along which fft is applied')
        if len(nffts) != 2:
            raise ValueError('provide at two nfft dimensions')
        if len(sampling) != 2:
            raise ValueError('provide two sampling steps')

        self.dirs = dirs
        self.nffts = np.array([int(nffts[0]) if nffts[0] is not None
                               else dims[self.dirs[0]],
                               int(nffts[1]) if nffts[1] is not None
                               else dims[self.dirs[1]]])
        self.f1 = np.fft.fftfreq(self.nffts[0], d=sampling[0])
        self.f2 = np.fft.fftfreq(self.nffts[1], d=sampling[1])

        self.dims = np.array(dims)
        self.dims_fft = self.dims.copy()
        self.dims_fft[self.dirs[0]] = self.nffts[0]
        self.dims_fft[self.dirs[1]] = self.nffts[1]

        self.shape = (int(np.prod(self.dims_fft)), int(np.prod(self.dims)))
        self.rdtype = np.dtype(dtype)
        self.cdtype = (np.ones(1, dtype=self.rdtype) +
                       1j * np.ones(1, dtype=self.rdtype)).dtype
        self.dtype = self.cdtype
        self.explicit = False

    def _matvec(self, x):
        x = np.reshape(x, self.dims)
        y = np.sqrt(1./np.prod(self.nffts)) * np.fft.fft2(x, s=self.nffts,
                                                          axes=(self.dirs[0],
                                                                self.dirs[1]))
        y = y.flatten()
        y = y.astype(self.cdtype)
        return y

    def _rmatvec(self, x):
        x = np.reshape(x, self.dims_fft)
        y = np.sqrt(np.prod(self.nffts)) * np.fft.ifft2(x, s=self.nffts,
                                                        axes=(self.dirs[0],
                                                              self.dirs[1]))
        y = np.take(y, range(self.dims[self.dirs[0]]), axis=self.dirs[0])
        y = np.take(y, range(self.dims[self.dirs[1]]), axis=self.dirs[1])
        y = y.flatten()
        y = y.astype(self.rdtype)
        return y
    