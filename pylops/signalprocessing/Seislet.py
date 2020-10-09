from math import log, ceil

import numpy as np
from pylops import LinearOperator
from pylops.basicoperators import Pad


def _predict_trace(trace, t, dt, dx, slope, adj=False):
    r"""Slope-based trace prediction.

    Resample a trace to a new time axis defined by the local slopes along the
    trace. Slopes do implicitly represent a time-varying time delay
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


def _predict_haar(traces, dt, dx, slopes, repeat=0, backward=False, adj=False):
    """Predict set of traces given time-varying slopes (Haar basis function)

    A set of input traces are resampled based on local slopes. If the number
    of traces in ``slopes`` is twice the number of traces in ``traces``, the
    resampling is done only once per trace. If the number of traces in
    ``slopes`` is a multiple of 2 of the number of traces in ``traces``,
    the prediction is done recursively or in other words the output traces
    are obtained by resampling the input traces followed by ``repeat-1``
    further resampling steps of the intermediate results.

    Parameters
    ----------
    traces : :obj:`numpy.ndarray`
        Input traces of size :math:`n_x \times n_t`
    dt : :obj:`float`
        Time axis sampling of the slope field
    dx : :obj:`float`
        Spatial axis sampling of the slope field
    slopes: :obj:`numpy.ndarray`
        Slope field of size :math:`n_x * 2^{repeat} \times n_t`
    repeat : :obj:`int`, optional
        Number of repeated predictions
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

    nx, nt = traces.shape
    t = np.arange(nt) * dt
    pred = np.zeros_like(traces)
    for ix in range(nx):
        pred_tmp = traces[ix]
        if adj:
            for irepeat in range(repeat - 1, -1, -1):
                #print('Slope at', ix * slopejump + iback * repeat + idir * irepeat)
                pred_tmp = \
                    _predict_trace(pred_tmp, t, dt, idir * dx,
                                   slopes[ix * slopejump + iback * repeat + idir * irepeat],
                                   adj=True)
        else:
            for irepeat in range(repeat):
                #print('Slope at', ix * slopejump + iback * repeat + idir * irepeat)
                pred_tmp = \
                    _predict_trace(pred_tmp, t, dt, idir * dx,
                                   slopes[ix * slopejump + iback * repeat + idir * irepeat])
        pred[ix] = pred_tmp
    return pred


def _predict_lin(traces, dt, dx, slopes, repeat=0, backward=False, adj=False):
    """Predict set of traces given time-varying slopes (Linear basis function)

    See _predict_haar for details.
    """
    if backward:
        iback = 1
        idir = -1
    else:
        iback = 0
        idir = 1
    slopejump = 2 ** (repeat + 1)
    repeat = 2 ** repeat

    nx, nt = traces.shape
    t = np.arange(nt) * dt
    pred = np.zeros_like(traces)
    for ix in range(nx):
        pred_tmp = traces[ix]
        #print('Data+ at', ix * slopejump)
        if adj:
            if not ((ix == 0 and not backward) or (ix == nx - 1 and backward)):
                pred_tmp1 = traces[ix - idir]
            #if ix > 0: print('Data- at', (ix - idir) * slopejump)
            for irepeat in range(repeat - 1, -1, -1):
                if (ix == 0 and not backward) or (ix == nx - 1 and backward):
                    #print('Slope+ at', ix * slopejump + iback * repeat + idir * irepeat)
                    pred_tmp = \
                        _predict_trace(pred_tmp, t, dt, idir * dx,
                                       slopes[ix * slopejump + iback * repeat + idir * irepeat],
                                       adj=True)
                    pred_tmp1 = 0
                else:
                    #print('Slope+ at', ix * slopejump + iback * repeat + idir * irepeat)
                    #print('Slope- at', ix * slopejump + iback * repeat - idir * irepeat)
                    pred_tmp = \
                        _predict_trace(pred_tmp, t, dt, idir * dx,
                                       slopes[ix * slopejump + iback * repeat + idir * irepeat],
                                       adj=True)
                    pred_tmp1 = \
                        _predict_trace(pred_tmp1, t, dt, (-idir) * dx,
                                       slopes[ix * slopejump + iback * repeat - idir * irepeat],
                                       adj=True)
        else:
            if not ((ix == nx - 1 and not backward) or (ix == 0 and backward)):
                pred_tmp1 = traces[ix + idir]
            #if ix < nx - 1: print('Data- at', (ix + idir) * slopejump)
            for irepeat in range(repeat):
                if (ix == nx - 1 and not backward) or (ix == 0 and backward):
                    #print('Slope+ at', ix * slopejump + iback * repeat + idir * irepeat)
                    pred_tmp = \
                        _predict_trace(pred_tmp, t, dt, idir * dx,
                                       slopes[ix * slopejump + iback * repeat + idir * irepeat])
                    pred_tmp1 = 0
                else:
                    #print('Slope+ at', ix * slopejump + iback * repeat + idir * irepeat)
                    #print('Slope- at', (ix + idir) * slopejump + iback * repeat - idir * irepeat)
                    pred_tmp = \
                        _predict_trace(pred_tmp, t, dt, idir * dx,
                                       slopes[ix * slopejump + iback * repeat + idir * irepeat])
                    pred_tmp1 = \
                        _predict_trace(pred_tmp1, t, dt, (-idir) * dx,
                                       slopes[(ix + idir) * slopejump + iback * repeat - idir * irepeat])

        #if (adj and ((ix == 0 and not backward) or (ix == nx - 1 and backward))) or
        #    (ix == nx - 1 and not backward) or (ix == 0 and backward):
        #    pred[ix] = pred_tmp
        #else:
        if ix == nx - 1:
            pred[ix] = pred_tmp + pred_tmp1 / 2.
        else:
            pred[ix] = (pred_tmp + pred_tmp1) / 2.
    return pred


class Seislet(LinearOperator):
    r"""Two dimensional Seislet operator.

    Apply 2D-Seislet Transform to an input array given an
    estimate of its local ``slopes``. In forward mode, the input array is
    reshaped into a two-dimensional array of size :math:`n_x \times n_t` and
    the transform is performed along the first (spatial) axis (see Notes for
    more details).

    Parameters
    ----------
    slopes : :obj:`numpy.ndarray`
        Slope field of size :math:`n_x \times n_t`
    sampling : :obj:`tuple`, optional
        Sampling steps in x- and t-axis.
    level : :obj:`int`, optional
        Number of scaling levels (must be >=0).
    kind : :obj:`str`, optional
        Basis function used for predict and update steps: ``haar`` or
        ``linear``.
    inv : :obj:`int`, optional
        Apply inverse transform when invoking the adjoint (``True``)
        or not (``False``). Note that in some scenario it may be more
        appropriate to use the exact inverse as adjoint of the Seislet
        operator even if this is not an orthogonal operator and the dot-test
        would not be satisfied (see Notes for details). Otherwise, the user
        can access the inverse directly as method of this class.
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
    NotImplementedError
        If ``kind`` is different from haar or linear
    ValueError
        If ``sampling`` has more or less than two elements.

    Notes
    -----
    The Seislet transform [1]_ is implemented using the lifting scheme.

    In its simplest form (i.e., corresponding to the Haar basis function for
    the Wavelet transform) the input dataset is separated into even
    (:math:`\mathbf{e}`) and odd (:math:`\mathbf{o}`) traces. Even traces are
    used to forward predict the odd traces using local slopes and the
    new odd traces (also referred to as residual) is defined as:

    .. math::
        \mathbf{o}^{i+1} = \mathbf{r}^i = \mathbf{o}^i - P(\mathbf{e}^i)

    where :math:`P = P^+` is the slope-based forward prediction operator
    (which is here implemented as a sinc-based resampling).
    The residual is then updated and summed to the even traces to obtain the
    new even traces (also referred to as coarse representation):

    .. math::
        \mathbf{e}^{i+1} = \mathbf{c}^i = \mathbf{e}^i + U(\mathbf{o}^{i+1})

    where :math:`U = P^- / 2` is the update operator which performs a
    slope-based backward prediction. At this point
    :math:`\mathbf{e}^{i+1}` becomes the new data and the procedure is repeated
    `level` times (at maximum until :math:`\mathbf{e}^{i+1}` is a single trace.
    The Seislet transform is effectively composed of all residuals and
    the coarsest data representation.

    In the inverse transform the two operations are reverted. Starting from the
    coarsest scale data representation :math:`\mathbf{c}` and residual
    :math:`\mathbf{r}`, the even and odd parts of the previous scale are
    reconstructed as:

    .. math::
        \mathbf{e}^i = \mathbf{c}^i - U(\mathbf{r}^i)
        = \mathbf{e}^{i+1} - U(\mathbf{o}^{i+1})

    and:

    .. math::
        \mathbf{o}^i  = \mathbf{r}^i + P(\mathbf{e}^i)
        = \mathbf{o}^{i+1} + P(\mathbf{e}^i)

    A new data is formed by interleaving :math:`\mathbf{e}^i` and
    :math:`\mathbf{o}^i` and the procedure repeated until the new data as the
    same number of traces as the original one.

    Finally the adjoint operator can be easily derived by writing the lifting
    scheme in a matricial form:

    .. math::
        \begin{bmatrix}
           \mathbf{r}_1  \\ \mathbf{r}_2  \\ ... \\ \mathbf{r}_N \\
           \mathbf{c}_1  \\ \mathbf{c}_2  \\ ... \\ \mathbf{c}_N
        \end{bmatrix} =
        \begin{bmatrix}
           \mathbf{I} & \mathbf{0} & ... & \mathbf{0} & -\mathbf{P} & \mathbf{0}  & ... & \mathbf{0}  \\
           \mathbf{0} & \mathbf{I} & ... & \mathbf{0} & \mathbf{0}  & -\mathbf{P} & ... & \mathbf{0}  \\
           ...        & ...        & ... & ...        & ...         & ...         & ... & ...         \\
           \mathbf{0} & \mathbf{0} & ... & \mathbf{I} & \mathbf{0}  & \mathbf{0}  & ... & -\mathbf{P} \\
           \mathbf{U} & \mathbf{0} & ... & \mathbf{0} & \mathbf{I}-\mathbf{UP} & \mathbf{0}  & ... & \mathbf{0}  \\
           \mathbf{0} & \mathbf{U} & ... & \mathbf{0} & \mathbf{0}  & \mathbf{I}-\mathbf{UP} & ... & \mathbf{0}  \\
           ...        & ...        & ... & ...        & ...         & ...         & ... & ...         \\
           \mathbf{0} & \mathbf{0} & ... & \mathbf{U} & \mathbf{0}  & \mathbf{0}  & ... & \mathbf{I}-\mathbf{UP} \\
        \end{bmatrix}
        \begin{bmatrix}
           \mathbf{o}_1  \\ \mathbf{o}_2  \\ ... \\ \mathbf{o}_N \\
           \mathbf{e}_1  \\ \mathbf{e}_2  \\ ... \\ \mathbf{e}_N \\
        \end{bmatrix}

    Transposing the operator leads to:

    .. math::
        \begin{bmatrix}
           \mathbf{o}_1  \\ \mathbf{o}_2  \\ ... \\ \mathbf{o}_N \\
           \mathbf{e}_1  \\ \mathbf{e}_2  \\ ... \\ \mathbf{e}_N \\
        \end{bmatrix} =
        \begin{bmatrix}
           \mathbf{I} & \mathbf{0} & ... & \mathbf{0} & -\mathbf{U^T} & \mathbf{0}  & ... & \mathbf{0}  \\
           \mathbf{0} & \mathbf{I} & ... & \mathbf{0} & \mathbf{0} & -\mathbf{U^T} & ... & \mathbf{0}  \\
           ...        & ...        & ... & ...        & ...        & ...        & ... & ...         \\
           \mathbf{0} & \mathbf{0} & ... & \mathbf{I} & \mathbf{0} & \mathbf{0} & ... & -\mathbf{U^T} \\
           \mathbf{P^T} & \mathbf{0} & ... & \mathbf{0} & \mathbf{I}-\mathbf{P^TU^T} & \mathbf{0} & ... & \mathbf{0}  \\
           \mathbf{0} & \mathbf{P^T} & ... & \mathbf{0} & \mathbf{0} & \mathbf{I}-\mathbf{P^TU^T} & ... & \mathbf{0}  \\
           ...        & ...        & ... & ...          & ...        & ...        & ... & ...         \\
           \mathbf{0} & \mathbf{0} & ... & \mathbf{P^T} & \mathbf{0} & \mathbf{0} & ... & \mathbf{I}-\mathbf{P^TU^T} \\
        \end{bmatrix}
        \begin{bmatrix}
           \mathbf{r}_1  \\ \mathbf{r}_2  \\ ... \\ \mathbf{r}_N \\
           \mathbf{c}_1  \\ \mathbf{c}_2  \\ ... \\ \mathbf{c}_N
        \end{bmatrix}

    which can be written more easily in the following two steps:

    .. math::
        \mathbf{o} = \mathbf{r} + \mathbf{U}^H\mathbf{c}

    and:

    .. math::
        \mathbf{e} = \mathbf{c} - \mathbf{P}^H(\mathbf{r} + \mathbf{U}^H(\mathbf{c})) =
                     \mathbf{c} - \mathbf{P}^H\mathbf{o}

    Similar derivations follow for more complex wavelet bases.

    .. [1] Fomel, S.,  Liu, Y., "Seislet transform and seislet frame",
       Geophysics, 75, no. 3, V25-V38. 2010.

    """
    def __init__(self, slopes, sampling=(1., 1.), level=None, kind='haar',
                 inv=False, dtype='float64'):
        if len(sampling) != 2:
            raise ValueError('provide two sampling steps')

        # define predict and update steps
        if kind == 'haar':
            self.predict = _predict_haar
        elif kind == 'linear':
            self.predict = _predict_lin
        else:
            raise NotImplementedError('kind should be haar or linear')

        # define padding for length to be power of 2
        dims = slopes.shape
        ndimpow2 = 2 ** ceil(log(dims[0], 2))
        pad = [(0, 0)] * len(dims)
        pad[0] = (0, ndimpow2 - dims[0])
        self.pad = Pad(dims, pad)
        self.dims = list(dims)
        self.dims[0] = ndimpow2
        self.nx, self.nt = self.dims

        # define levels
        nlevels_max = int(np.log2(self.dims[0]))
        self.levels_size = np.flip(np.array([2 ** i for i in range(nlevels_max)]))
        if level is not None:
            self.levels_size = self.levels_size[:level]
        else:
            self.levels_size = self.levels_size[:-1]
            level = nlevels_max - 1
        self.level = level
        self.levels_cum = np.cumsum(self.levels_size)
        self.levels_cum = np.insert(self.levels_cum, 0, 0)

        self.dx, self.dt = sampling
        self.slopes = (self.pad * slopes.ravel()).reshape(self.dims)
        self.inv = inv
        self.shape = (int(np.prod(self.slopes.size)),
                      int(np.prod(slopes.size)))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = self.pad.matvec(x)
        x = np.reshape(x, self.dims)
        y = np.zeros((np.sum(self.levels_size) + self.levels_size[-1], self.nt))
        for ilevel in range(self.level):
            odd = x[1::2]
            even = x[::2]
            res = odd - self.predict(even, self.dt, self.dx, self.slopes,
                                     repeat=ilevel, backward=False)
            x = even + self.predict(res, self.dt, self.dx,
                                    self.slopes, repeat=ilevel,
                                    backward=True) / 2.
            y[self.levels_cum[ilevel]:self.levels_cum[ilevel + 1]] = res
        y[self.levels_cum[-1]:] = x
        return y.ravel()

    def _rmatvec(self, x):
        if not self.inv:
            x = np.reshape(x, self.dims)
            y = x[self.levels_cum[-1]:]
            for ilevel in range(self.level, 0, -1):
                res = x[self.levels_cum[ilevel - 1]:self.levels_cum[ilevel]]
                odd = res + self.predict(y, self.dt, self.dx, self.slopes,
                                         repeat=ilevel - 1, backward=True,
                                         adj=True) / 2.
                even = y - self.predict(odd, self.dt, self.dx, self.slopes,
                                        repeat=ilevel - 1, backward=False,
                                        adj=True)
                y = np.zeros((2 * even.shape[0], self.nt))
                y[1::2] = odd
                y[::2] = even
            y = self.pad.rmatvec(y.ravel())
        else:
            y = self.inverse(x)
        return y

    def inverse(self, x):
        x = np.reshape(x, self.dims)
        y = x[self.levels_cum[-1]:]
        for ilevel in range(self.level, 0, -1):
            res = x[self.levels_cum[ilevel - 1]:self.levels_cum[ilevel]]
            even = y - self.predict(res, self.dt, self.dx, self.slopes,
                                    repeat=ilevel - 1, backward=True) / 2.
            odd = res + self.predict(even, self.dt, self.dx, self.slopes,
                                     repeat=ilevel - 1, backward=False)
            y = np.zeros((2 * even.shape[0], self.nt))
            y[1::2] = odd
            y[::2] = even
        y = self.pad.rmatvec(y.ravel())
        return y
