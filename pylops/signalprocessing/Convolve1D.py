import numpy as np
from scipy.signal import convolve, correlate, fftconvolve
from pylops import LinearOperator


class Convolve1D(LinearOperator):
    r"""1D convolution operator.

    Apply one-dimensional convolution with a compact filter to model (and data)
    along a specific direction of a multi-dimensional array depending on the
    choice of ``dir``.

    Parameters
    ----------
    N : :obj:`int`
        Number of samples in model.
    h : :obj:`numpy.ndarray`
        1d compact filter to be convolved to input signal
    offset : :obj:`int`
        Index of the center of the compact filter
    dims : :obj:`tuple`
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which convolution is applied
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    Notes
    -----
    The Convolve1D operator applies convolution between the input signal
    :math:`x(t)` and a compact filter kernel :math:`h(t)` in forward model:

    .. math::
        y(t) = \int_{-\inf}^{\inf} h(t-\tau) x(\tau) d\tau

    This operation can be discretized as follows

    .. math::
        y[n] = \sum_{m=-\inf}^{\inf} h[n-m] x[m]

    as well as performed in the frequency domain.

    .. math::
        Y(f) = \mathscr{F} (h(t)) * \mathscr{F} (x(t))

    Convolve1D operator uses :py:func:`scipy.signal.convolve` that
    automatically chooses the best domain for the operation to be carried out
    for one dimensional inputs. The fft implementation
    :py:func:`scipy.signal.fftconvolve` is however enforced for signals in
    2 or more dimensions as this routine efficently operates on
    multi-dimensional arrays.

    As the adjoint of convolution is correlation, Convolve1D operator applies
    correlation in the adjoint mode.

    In time domain:

    .. math::
        x(t) = \int_{-\inf}^{\inf} h(t+\tau) x(\tau) d\tau

    or in frequency domain:

    .. math::
        y(t) = \mathscr{F}^{-1} (H(f)^* * X(f))

    """
    def __init__(self, N, h, offset=0, dims=None, dir=0, dtype='float64'):
        self.offset = int(offset)
        self.h = h
        self.hstar = np.flip(h)
        if dims is not None:
            # add dimensions to filter to match dimensions of model and data
            hdims = [1] * len(dims)
            hdims[dir] = len(h)
            self.h = self.h.reshape(hdims)
            self.hstar = self.hstar.reshape(hdims)
        self.dir = dir
        if dims is None:
            self.dims = np.array([N, 1])
            self.reshape = False
        else:
            if np.prod(dims) != N:
                raise ValueError('product of dims must equal N!')
            else:
                self.dims = np.array(dims)
                self.reshape = True
        self.shape = (np.prod(self.dims), np.prod(self.dims))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if not self.reshape:
            y = convolve(x, self.h, mode='full')
            y = y[self.offset:-self.h.size + self.offset + 1]
        else:
            x = np.reshape(x, self.dims)
            y = fftconvolve(x, self.h, mode='full', axes=self.dir)
            y = np.take(y, np.arange(self.offset, y.shape[self.dir] -
                                     (self.h.size - self.offset - 1)),
                        axis=self.dir)
            y = y.ravel()
        return y

    def _rmatvec(self, x):
        if not self.reshape:
            y = correlate(x, self.h, mode='full')
            y = y[self.h.size - self.offset - 1:y.size-self.offset]
        else:
            x = np.reshape(x, self.dims)
            y = fftconvolve(x, self.hstar, mode='full', axes=self.dir)
            y = np.take(y, np.arange(self.h.size - self.offset - 1,
                                     y.shape[self.dir] - self.offset),
                        axis=self.dir)
            y = y.ravel()
        return y
