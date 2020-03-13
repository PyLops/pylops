from math import log, ceil

import numpy as np
from pylops import LinearOperator
from pylops.basicoperators import Pad


def _predict_trace(trace, t, dt, dx, slope, adj=False):
    r"""Slope-based trace prediction.

    Resample a trace to a new time axis defined by the local slopes along the
    trace. Slopes do implicitely represent a time-varying time delay
    :math:`\Delta t (t) = dx*s(t)`.

    The input trace is interpolated using sinc-interpolation to a new time
    axis given by the following formula: :math:`t_{new} = t + dx*s(t)`.

    Parameters
    ----------
    trace : :obj:`numpy.ndarray`
        Trace
    t : :obj:`numpy.ndarray`
        Time axis
    dt : :obj:`float`
        Time axis sampling
    dx : :obj:`float`
        Spatial axis sampling
    slope : :obj:`numpy.ndarray`
        Slope field
    adj : :obj:`bool`, optional
        Perform forward (``False``) or adjoint (``True``) operation

    Returns
    -------
    tracenew : :obj:`numpy.ndarray`
        Resampled trace

    """
    newt = t - dx * slope
    sinc = np.tile(newt, (len(newt), 1)) - \
           np.tile(t[:, np.newaxis], (1, len(newt)))
    if adj:
        tracenew = np.dot(trace, np.sinc(sinc / dt).T)
    else:
        tracenew = np.dot(trace, np.sinc(sinc / dt))
    return tracenew


def _predict(traces, dt, dx, slopes, repeat=0,
             backward=False, adj=False):
    """Predict set of traces given time-varying slopes.

    A set of input traces are resampled based on local slopes. If ``traces``
    are by multiples spatial steps ``dx``, the prediction is done recursively
    or in other words the output traces are obtained by resampling the input
    traces followed by ``repeat-1`` further resampling steps of the
    intermediate results. Note that local slopes must be always provided
    at ``dx`` sampling.

    Parameters
    ----------
    traces : :obj:`numpy.ndarray`
        Input traces of size :math:`n_t \times n_x`
    dt : :obj:`float`
        Time axis sampling
    dx : :obj:`float`
        Spatial axis sampling
    slopes: :obj:`numpy.ndarray`
        Slope field of size :math:`n_t \times n_x* 2^{repeat}`
    backward : :obj:`bool`, optional
        Predicted trace is on the right (``False``) or on the left (``True``)
        of input trace
    adj : :obj:`bool`, optional
        Perform forward (``False``) or adjoint (``True``) operation

    Returns
    -------
    pred : :obj:`numpy.ndarray`
        Predicted traces

    """
    if backward:
        iback = 1
        idir = -1
    else:
        iback = 0
        idir = 1
    slopejump = 2 ** (repeat + 1)
    repeat = 2 ** repeat

    nt, nx = traces.shape
    t = np.arange(nt) * dt
    pred = np.zeros_like(traces)
    for ix in range(nx):
        pred_tmp = traces[:, ix]
        if adj:
            for irepeat in range(repeat - 1, -1, -1):
                pred_tmp = \
                    _predict_trace(pred_tmp, t, dt, idir * dx,
                                   slopes[:, ix * slopejump + iback * repeat + idir * irepeat],
                                   adj=True)
        else:
            for irepeat in range(repeat):
                pred_tmp = \
                    _predict_trace(pred_tmp, t, dt, idir * dx,
                                   slopes[:, ix*slopejump + iback * repeat + idir * irepeat])
        pred[:, ix] = pred_tmp
    return pred


class Seislet(LinearOperator):
    r"""Two dimensional Seislet operator.

    Apply 2D-Seislet Transform to a two-dimensional input dataset given an
    estimate of its local ``slopes``.

    Parameters
    ----------
    slopes: :obj:`numpy.ndarray`
        Slope field
    sampling : :obj:`tuple`, optional
        Sampling steps ``dy`` and ``dx``
    level : :obj:`int`, optional
        Number of scaling levels (must be >=0).
    dtype : :obj:`str`, optional
        Type of elements in input array.

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
        If ``sampling`` has more or less than two elements.

    Notes
    -----
    The Seislet transform [1]_ is implemented using the lifting scheme.

    In its simplest form (i.e., corresponding to the Haar basis function for
    the Wavelet transform) the input dataset is separated into even
    (:math:`\mathbf{e}`) and odd (:math:`\mathbf{o}`) traces. Even traces are
    used to forward predict the odd traces using local slopes and the residual
    is defined as:

    .. math::
        \mathbf{r} = \mathbf{o} - P(\mathbf{e})

    where :math:`P` is the slope-based prediction operator (which is here
    implemented as a sinc-based resampling). The residual is then updated
    and summed to the even traces:

    .. math::
        \mathbf{c} = \mathbf{e} + U(\mathbf{r})

    where :math:`U = P / 2` is the update operator. At this point
    :math:`\mathbf{c}` becomes the new data and the procedure is repeated
    `level` times (at maximum until :math:`\mathbf{c}` is a single trace. The
    Seislet transform is effectively composed of all residuals and
    the coarsest data representation.

    In the inverse transform the two operations are reverted. Starting from the
    coarsest scale data representation :math:`\mathbf{c}` and residual
    :math:`\mathbf{r}`, the even and odd parts of the previous scale are
    reconstructed as:

    .. math::
        \mathbf{e} = \mathbf{c} - U(\mathbf{r})

    and:

    .. math::
        \mathbf{o} = \mathbf{r} + P(\mathbf{e})

    A new data is formed and the procedure repeated until the new data as the
    same number of traces as the original one.

    Finally the adjoint operator can be easily derived by writing the lifting
    scheme in a matricial form:

    .. math::
        \begin{bmatrix}
           \mathbf{r}_1  \\ \mathbf{r}_2  \\ ...  \\
           \mathbf{c}_1  \\ \mathbf{c}_2  \\ ...
        \end{bmatrix} =
        \begin{bmatrix}
           -\mathbf{P} & \mathbf{I} & \mathbf{0} & \mathbf{0} & ...  & \mathbf{0} & \mathbf{0} \\
           \mathbf{0} & \mathbf{0} & -\mathbf{P} & \mathbf{I} & ...  & \mathbf{0} & \mathbf{0} \\
           ... & ... & ...  & ... & ...  & ... & ... \\
           \mathbf{I}-\mathbf{UP} & \mathbf{U} & \mathbf{0} & \mathbf{0} & ...  & \mathbf{0} & \mathbf{0} \\
           \mathbf{0} & \mathbf{0} & \mathbf{I}-\mathbf{UP} & U & ...  & \mathbf{0} & \mathbf{0} \\
           ... & ... & ...  & ... & ...  & ... & ...
        \end{bmatrix}
        \begin{bmatrix}
           \mathbf{e}_1  \\ \mathbf{o}_1  \\ \mathbf{e}_2 \\ \mathbf{o}_2  \\
           ...  \\ \mathbf{e}_N \\ \mathbf{o}_N
        \end{bmatrix}

    Transposing the operator leads to:

    .. math::
        \begin{bmatrix}
           \mathbf{e}_1  \\ \mathbf{o}_1  \\ \mathbf{e}_2 \\ \mathbf{o}_2  \\
           ...  \\ \mathbf{e}_N \\ \mathbf{o}_N
        \end{bmatrix} =
        \begin{bmatrix}
           -\mathbf{P}^H & \mathbf{0} & ... & \mathbf{I}-\mathbf{P}^H\mathbf{U}^H  & \mathbf{0} & ... \\
           \mathbf{I} & \mathbf{0} & ... & \mathbf{U}^H  & \mathbf{0} & ... \\
           \mathbf{0} & -\mathbf{P}^H & ... & \mathbf{0} & \mathbf{I}-\mathbf{P}^H\mathbf{U}^H  & ...\\
           \mathbf{0} & \mathbf{I} & ... & \mathbf{0} & \mathbf{U}^H  & ...\\
           ... & ... & ...  & ... & ...  & ... \\
        \end{bmatrix}
        \begin{bmatrix}
           \mathbf{r}_1  \\ \mathbf{r}_2  \\ ...  \\
           \mathbf{c}_1  \\ \mathbf{c}_2  \\ ...
        \end{bmatrix}

    which can written more easily in the following two steps:

    .. math::
        \mathbf{o} = \mathbf{r} - \mathbf{U}^H\mathbf{c}

    and:

    .. math::
        \mathbf{e} = \mathbf{c} - \mathbf{P}^H(\mathbf{r} - \mathbf{U}^H(\mathbf{c})) =
                     \mathbf{c} - \mathbf{P}^H\mathbf{o}

    .. [1] Fomel, S.,  Liu, Y., "Seislet transform and seislet frame",
       Geophysics, 75, no. 3, V25-V38. 2010.

    """
    def __init__(self, slopes, sampling=(1., 1.),
                 level=None, dtype='float64'):
        # checks
        if len(sampling) != 2:
            raise ValueError('provide two sampling steps')

        # define padding for length to be power of 2
        dims = slopes.shape
        ndimpow2 = 2 ** ceil(log(dims[1], 2))
        pad = [(0, 0)] * len(dims)
        pad[1] = (0, ndimpow2 - dims[1])
        self.pad = Pad(dims, pad)
        self.dims = list(dims)
        self.dims[1] = ndimpow2
        self.nt, self.nx = self.dims[0], self.dims[1]

        # define levels
        nlevels_max = int(np.log2(dims[1]))
        self.levels_size = np.flip(np.array([2 ** i for i in range(nlevels_max)]))
        if level is not None:
            self.levels_size = self.levels_size[:level]
        else:
            self.levels_size = self.levels_size[:-1]
            level = nlevels_max - 1
        self.level = level
        self.levels_cum = np.cumsum(self.levels_size)
        self.levels_cum = np.insert(self.levels_cum, 0, 0)
        self.dt, self.dx = sampling[0], sampling[1]
        self.slopes = (self.pad * slopes.ravel()).reshape(self.dims)
        self.shape = (int(np.prod(self.dims)), int(np.prod(self.dims)))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = self.pad.matvec(x)
        x = np.reshape(x, self.dims)
        y = np.zeros((self.nt, np.sum(self.levels_size) + self.levels_size[-1]))
        for ilevel in range(self.level):
            odd = x[:, 1::2]
            even = x[:, ::2]
            res = odd - _predict(even, self.dt, self.dx, self.slopes,
                                 repeat=ilevel, backward=False)
            x = even + _predict(res, self.dt, self.dx,
                                self.slopes, repeat=ilevel,
                                backward=True) / 2.
            y[:, self.levels_cum[ilevel]:self.levels_cum[ilevel + 1]] = res
        y[:, self.levels_cum[-1]:] = x
        return y.ravel()

    def _rmatvec(self, x):
        x = np.reshape(x, self.dims)
        y = x[:, self.levels_cum[-1]:]
        for ilevel in range(self.level, 0, -1):
            res = x[:, self.levels_cum[ilevel - 1]:self.levels_cum[ilevel]]
            odd = res + _predict(y, self.dt, self.dx, self.slopes,
                                 repeat=ilevel - 1, backward=True,
                                 adj=True) / 2.
            even = y - _predict(odd, self.dt, self.dx, self.slopes,
                                repeat=ilevel - 1, backward=False, adj=True)
            y = np.zeros((self.nt, 2 * even.shape[1]))
            y[:, 1::2] = odd
            y[:, ::2] = even
        y = self.pad.rmatvec(y.ravel())
        return y

    def inverse(self, x):
        x = np.reshape(x, self.dims)
        y = x[:, self.levels_cum[-1]:]
        for ilevel in range(self.level, 0, -1):
            res = x[:, self.levels_cum[ilevel - 1]:self.levels_cum[ilevel]]
            even = y - _predict(res, self.dt, self.dx, self.slopes,
                                repeat=ilevel - 1, backward=True) / 2.
            odd = res + _predict(even, self.dt, self.dx, self.slopes,
                                 repeat=ilevel - 1, backward=False)
            y = np.zeros((self.nt, 2 * even.shape[1]))
            y[:, 1::2] = odd
            y[:, ::2] = even
        y = self.pad.rmatvec(y.ravel())
        return y
