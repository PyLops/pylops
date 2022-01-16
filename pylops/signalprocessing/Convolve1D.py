import numpy as np

from pylops import LinearOperator
from pylops.utils.backend import (
    get_convolve,
    get_fftconvolve,
    get_oaconvolve,
    to_cupy_conditional,
)


def _choose_convfunc(x, method, dims):
    """Choose convolution function

    Choose and return the function handle to be used for convolution
    """
    if dims is None:
        if method is None:
            method = "direct"
        if method not in ("direct", "fft"):
            raise NotImplementedError("method must be direct or fft")
        convfunc = get_convolve(x)
    else:
        if method is None:
            method = "fft"
        if method == "fft":
            convfunc = get_fftconvolve(x)
        elif method == "overlapadd":
            convfunc = get_oaconvolve(x)
        else:
            raise NotImplementedError("method must be fft or overlapadd")
    return convfunc, method


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
    method : :obj:`str`, optional
        Method used to calculate the convolution (``direct``, ``fft``,
        or ``overlapadd``). Note that only ``direct`` and ``fft`` are allowed
        when ``dims=None``, whilst ``fft`` and ``overlapadd`` are allowed
        when ``dims`` is provided.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    Raises
    ------
    ValueError
        If ``offset`` is bigger than ``len(h) - 1``
    NotImplementedError
        If ``method`` provided is not allowed

    Notes
    -----
    The Convolve1D operator applies convolution between the input signal
    :math:`x(t)` and a compact filter kernel :math:`h(t)` in forward model:

    .. math::
        y(t) = \int\limits_{-\infty}^{\infty} h(t-\tau) x(\tau) \,\mathrm{d}\tau

    This operation can be discretized as follows

    .. math::
        y[n] = \sum_{m=-\infty}^{\infty} h[n-m] x[m]

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
        x(t) = \int\limits_{-\infty}^{\infty} h(t+\tau) x(\tau) \,\mathrm{d}\tau

    or in frequency domain:

    .. math::
        y(t) = \mathscr{F}^{-1} (H(f)^* * X(f))

    """

    def __init__(self, N, h, offset=0, dims=None, dir=0, dtype="float64", method=None):
        if offset > len(h) - 1:
            raise ValueError("offset must be smaller than len(h) - 1")
        self.h = h
        self.hstar = np.flip(self.h, axis=-1)
        self.nh = len(h)
        self.offset = 2 * (self.nh // 2 - int(offset))
        if self.nh % 2 == 0:
            self.offset -= 1
        if self.offset != 0:
            self.h = np.pad(
                self.h,
                (
                    self.offset if self.offset > 0 else 0,
                    -self.offset if self.offset < 0 else 0,
                ),
                mode="constant",
            )
        self.hstar = np.flip(self.h, axis=-1)
        self.dimsorig = dims
        if dims is not None:
            # add dimensions to filter to match dimensions of model and data
            hdims = [1] * len(dims)
            hdims[dir] = len(self.h)
            self.h = self.h.reshape(hdims)
            self.hstar = self.hstar.reshape(hdims)
        self.dir = dir
        if dims is None:
            self.dims = np.array([N, 1])
            self.reshape = False
        else:
            if np.prod(dims) != N:
                raise ValueError("product of dims must equal N!")
            else:
                self.dims = np.array(dims)
                self.reshape = True
        # choose method and function handle
        self.convfunc, self.method = _choose_convfunc(h, method, self.dimsorig)
        self.shape = (np.prod(self.dims), np.prod(self.dims))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        if type(self.h) != type(x):
            self.h = to_cupy_conditional(x, self.h)
            self.convfunc, self.method = _choose_convfunc(
                self.h, self.method, self.dimsorig
            )
        if not self.reshape:
            y = self.convfunc(x.squeeze(), self.h, mode="same", method=self.method)
        else:
            x = np.reshape(x, self.dims)
            y = self.convfunc(x, self.h, mode="same")
            y = y.ravel()
        return y

    def _rmatvec(self, x):
        if type(self.hstar) != type(x):
            self.hstar = to_cupy_conditional(x, self.hstar)
            self.convfunc, self.method = _choose_convfunc(
                self.hstar, self.method, self.dimsorig
            )
        if not self.reshape:
            y = self.convfunc(x.squeeze(), self.hstar, mode="same", method=self.method)
        else:
            x = np.reshape(x, self.dims)
            y = self.convfunc(x, self.hstar, mode="same")
            y = y.ravel()
        return y
