import logging
import numpy as np
from pylops.basicoperators import Spread

try:
    from numba import jit
    from ._Radon3D_numba import _linear_numba, _parabolic_numba, \
        _hyperbolic_numba, _indices_3d_numba, _indices_3d_onthefly_numba, \
        _create_table_numba
except ModuleNotFoundError:
    jit = None

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def _linear(y, x, t, py, px):
    return t + px*x + py*y

def _parabolic(y, x, t, py, px):
    return t + px*x**2 + py*y**2

def _hyperbolic(y, x, t, py, px):
    return np.sqrt(t**2 + (x/px)**2 + (y/py)**2)

def _indices_3d(f, y, x, py, px, t, nt, interp=True):
    """Compute time and space indices of parametric line in ``f`` function

    Parameters
    ----------
    f : :obj:`func`
        Function computing values of parametric line for stacking
    y : :obj:`np.ndarray`
        Slow spatial axis (must be symmetrical around 0 and with sampling 1)
    x : :obj:`np.ndarray`
        Fast spatial axis (must be symmetrical around 0 and with sampling 1)
    py : :obj:`float`
        Slowness/curvature in slow axis
    px : :obj:`float`
        Slowness/curvature in fast axis
    t : :obj:`int`
        Time sample (time axis is assumed to have sampling 1)
    nt : :obj:`int`
        Size scaof time axis
    interp : :obj:`bool`, optional
        Apply linear interpolation (``True``) or nearest interpolation
        (``False``) during stacking/spreading along parametric curve

    Returns
    -------
    sscan : :obj:`np.ndarray`
        Spatial indices
    tscan : :obj:`np.ndarray`
        Time indices
    dtscan : :obj:`np.ndarray`
        Decimal time variations for interpolation

    """
    tdecscan = f(y, x, t, py, px)
    if not interp:
        sscan = (tdecscan >= 0) & (tdecscan < nt)
    else:
        sscan = (tdecscan >= 0) & (tdecscan < nt - 1)
    tscan = tdecscan[sscan].astype(np.int)
    if interp:
        dtscan = tdecscan[sscan] - tscan
    else:
        dtscan = None
    return sscan, tscan, dtscan

def _indices_3d_onthefly(f, y, x, py, px, ip, it, nt, interp=True):
    """Wrapper around _indices_3d to allow on-the-fly computation of
    parametric curves"""
    tscan = np.full(len(y), np.nan, dtype=np.float32)
    if interp:
        dtscan = np.full(len(y), np.nan)
    else:
        dtscan = None
    sscan, tscan1, dtscan1 = \
        _indices_3d(f, y, x, py[ip], px[ip], it, nt, interp=interp)
    tscan[sscan] = tscan1
    if interp:
        dtscan[sscan] = dtscan1
    return sscan, tscan, dtscan

def _create_table(f, y, x, pyaxis, pxaxis, nt, npy, npx, ny, nx, interp):
    """Create look up table
    """
    table = np.full((npx * npy, nt, ny * nx), np.nan, dtype=np.float32)
    if interp:
        dtable = np.full((npx * npy, nt, ny * nx), np.nan)
    else:
        dtable = None

    for ip, (py, px) in enumerate(zip(pyaxis, pxaxis)):
        for it in range(nt):
            sscan, tscan, dtscan = _indices_3d(f, y, x,
                                               py, px,
                                               it, nt,
                                               interp=interp)
            table[ip, it, sscan] = tscan
            if interp:
                dtable[ip, it, sscan] = dtscan
    return table, dtable


def Radon3D(taxis, hyaxis, hxaxis, pyaxis, pxaxis, kind='linear',
            centeredh=True, interp=True, onthefly=False,
            engine='numpy', dtype='float64'):
    r"""Three dimensional Radon transform.

    Apply three dimensional Radon forward (and adjoint) transform to a
    3-dimensional array of size :math:`[n_{py} \times n_{px} \times n_t]`
    (and :math:`[n_y \times n_x \times n_t]`).

    In forward mode this entails to spreading the model vector
    along parametric curves (lines, parabolas, or hyperbolas depending on the
    choice of ``kind``), while  stacking values in the data vector
    along the same parametric curves is performed in adjoint mode.

    Parameters
    ----------
    taxis : :obj:`np.ndarray`
        Time axis
    hxaxis : :obj:`np.ndarray`
        Fast patial axis
    hyaxis : :obj:`np.ndarray`
        Slow spatial axis
    pyaxis : :obj:`np.ndarray`
        Axis of scanning variable :math:`p_y` of parametric curve
    pxaxis : :obj:`np.ndarray`
        Axis of scanning variable :math:`p_x` of parametric curve
    kind : :obj:`str`, optional
        Curve to be used for stacking/spreading (``linear``, ``parabolic``,
        and ``hyperbolic`` are currently supported)
    centeredh : :obj:`bool`, optional
        Assume centered spatial axis (``True``) or not (``False``)
    interp : :obj:`bool`, optional
        Apply linear interpolation (``True``) or nearest interpolation
        (``False``) during stacking/spreading along parametric curve
    onthefly : :obj:`bool`, optional
        Compute stacking parametric curves on-the-fly as part of forward
        and adjoint modelling (``True``) or at initialization and store them
        in look-up table (``False``). Using a look-up table is computationally
        more efficient but increases the memory burden
    engine : :obj:`str`, optional
        Engine used for computation (``numpy`` or ``numba``)
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Returns
    -------
    r3op : :obj:`pylops.LinearOperator`
        Radon operator

    Raises
    ------
    KeyError
        If ``engine`` is neither ``numpy`` nor ``numba``
    NotImplementedError
        If ``kind`` is not ``linear``, ``parabolic``, or ``hyperbolic``

    See Also
    --------
    pylops.signalprocessing.Radon2D: Two dimensional Radon transform
    pylops.Spread: Spread operator

    Notes
    -----
    The Radon3D operator applies the following linear transform in adjoint mode
    to the data after reshaping it into a 3-dimensional array of
    size :math:`[n_y \times n_x \times n_t]` in adjoint mode:

    .. math::
        m(p_y, p_x, t_0) = \int{d(y, x, t = f(p_y, p_x, y, x, t))} dx dy

    where :math:`f(p_y, p_x, y, x, t) = t_0 + p_y * y + p_x * x` in linear
    mode, :math:`f(p_y, p_x, y, x, t) = t_0 + p_y * y^2 + p_x * x^2` in
    parabolic mode, and
    :math:`f(p_y, p_x, y, x, t) = \sqrt{t_0^2 + y^2 / p_y^2 + x^2 / p_x^2}`
    in hyperbolic mode.

    As the adjoint operator can be interpreted as a repeated summation of sets
    of elements of the model vector along chosen parametric curves, the
    forward is implemented as spreading of values in the data vector along the
    same parametric curves. This operator is actually a thin wrapper around
    the :class:`pylops.Spread` operator.
    """
    # engine
    if not engine in ['numpy', 'numba']:
        raise KeyError('engine must be numpy or numba')
    if engine == 'numba' and jit is None:
        engine = 'numpy'

    # axes
    nt, nhy, nhx = taxis.size, hyaxis.size, hxaxis.size
    npy, npx = pyaxis.size, pxaxis.size
    if kind == 'linear':
        f = _linear if engine == 'numpy' else _linear_numba
    elif kind == 'parabolic':
        f = _parabolic if engine == 'numpy' else _parabolic_numba
    elif kind == 'hyperbolic':
        f = _hyperbolic if engine == 'numpy' else _hyperbolic_numba
    else:
        raise NotImplementedError('kind must be linear, '
                                  'parabolic, or hyperbolic...')
    # make axes unitless
    dpy = (np.abs(hyaxis[1] - hyaxis[0]) /
           np.abs(taxis[1] - taxis[0]))
    pyaxis = pyaxis * dpy
    hyaxisunitless = np.arange(nhy)
    dpx = (np.abs(hxaxis[1] - hxaxis[0]) /
           np.abs(taxis[1] - taxis[0]))
    pxaxis = pxaxis * dpx
    hxaxisunitless = np.arange(nhx)
    if centeredh:
        hyaxisunitless -= nhy // 2
        hxaxisunitless -= nhx // 2

    # create grid for py and px axis
    hyaxisunitless, hxaxisunitless = \
        np.meshgrid(hyaxisunitless, hxaxisunitless, indexing='ij')
    pyaxis, pxaxis = np.meshgrid(pyaxis, pxaxis, indexing='ij')

    dims = (npy*npx, nt)
    dimsd = (nhy*nhx, nt)

    if onthefly:
        if engine == 'numba':
            @jit(nopython=True, nogil=True)
            def ontheflyfunc(x, y):
                return _indices_3d_onthefly_numba(f, hyaxisunitless.ravel(),
                                                  hxaxisunitless.ravel(),
                                                  pyaxis.ravel(),
                                                  pxaxis.ravel(),
                                                  x, y, nt, interp=interp)[1:]
        else:
            if interp:
                ontheflyfunc = \
                    lambda x, y: _indices_3d_onthefly(f,
                                                      hyaxisunitless.ravel(),
                                                      hxaxisunitless.ravel(),
                                                      pyaxis.ravel(),
                                                      pxaxis.ravel(),
                                                      x, y, nt, interp=interp)[1:]
            else:
                ontheflyfunc = \
                    lambda x, y: _indices_3d_onthefly(f,
                                                      hyaxisunitless.ravel(),
                                                      hxaxisunitless.ravel(),
                                                      pyaxis.ravel(),
                                                      pxaxis.ravel(),
                                                      x, y, nt, interp=interp)[1]
        r3op = Spread(dims, dimsd, fh=ontheflyfunc, interp=interp,
                      engine=engine, dtype=dtype)
    else:
        if engine == 'numba':
            tablefunc = _create_table_numba
        else:
            tablefunc = _create_table

        table, dtable = tablefunc(f, hyaxisunitless.ravel(),
                                  hxaxisunitless.ravel(),
                                  pyaxis.ravel(), pxaxis.ravel(),
                                  nt, npy, npx, nhy, nhx, interp=interp)
        if not interp:
            dtable = None
        r3op = Spread(dims, dimsd, table=table,
                      dtable=dtable, interp=interp,
                      engine=engine, dtype=dtype)
    return r3op
