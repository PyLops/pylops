import logging

import numpy as np

from pylops.basicoperators import Spread

try:
    from numba import jit

    from ._Radon2D_numba import (
        _create_table_numba,
        _hyperbolic_numba,
        _indices_2d_onthefly_numba,
        _linear_numba,
        _parabolic_numba,
    )
except ModuleNotFoundError:
    jit = None

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def _linear(x, t, px):
    return t + px * x


def _parabolic(x, t, px):
    return t + px * x ** 2


def _hyperbolic(x, t, px):
    return np.sqrt(t ** 2 + (x / px) ** 2)


def _indices_2d(f, x, px, t, nt, interp=True):
    """Compute time and space indices of parametric line in ``f`` function

    Parameters
    ----------
    f : :obj:`func`
        Function computing values of parametric line for stacking
    x : :obj:`np.ndarray`
        Spatial axis (must be symmetrical around 0 and with sampling 1)
    px : :obj:`float`
        Slowness/curvature
    t : :obj:`int`
        Time sample (time axis is assumed to have sampling 1)
    nt : :obj:`int`
        Size of time axis
    interp : :obj:`bool`, optional
        Apply linear interpolation (``True``) or nearest interpolation
        (``False``) during stacking/spreading along parametric curve

    Returns
    -------
    xscan : :obj:`np.ndarray`
        Spatial indices
    tscan : :obj:`np.ndarray`
        Time indices
    dtscan : :obj:`np.ndarray`
        Decimal time variations for interpolation

    """
    tdecscan = f(x, t, px)
    if not interp:
        xscan = (tdecscan >= 0) & (tdecscan < nt)
    else:
        xscan = (tdecscan >= 0) & (tdecscan < nt - 1)
    tscan = tdecscan[xscan].astype(int)
    if interp:
        dtscan = tdecscan[xscan] - tscan
    else:
        dtscan = None
    return xscan, tscan, dtscan


def _indices_2d_onthefly(f, x, px, ip, t, nt, interp=True):
    """Wrapper around _indices_2d to allow on-the-fly computation of
    parametric curves"""
    tscan = np.full(len(x), np.nan, dtype=np.float32)
    if interp:
        dtscan = np.full(len(x), np.nan)
    else:
        dtscan = None
    xscan, tscan1, dtscan1 = _indices_2d(f, x, px[ip], t, nt, interp=interp)
    tscan[xscan] = tscan1
    if interp:
        dtscan[xscan] = dtscan1
    return xscan, tscan, dtscan


def _create_table(f, x, pxaxis, nt, npx, nx, interp):
    """Create look up table"""
    table = np.full((npx, nt, nx), np.nan, dtype=np.float32)
    if interp:
        dtable = np.full((npx, nt, nx), np.nan)
    else:
        dtable = None

    for ipx, px in enumerate(pxaxis):
        for it in range(nt):
            xscan, tscan, dtscan = _indices_2d(f, x, px, it, nt, interp=interp)
            table[ipx, it, xscan] = tscan
            if interp:
                dtable[ipx, it, xscan] = dtscan
    return table, dtable


def Radon2D(
    taxis,
    haxis,
    pxaxis,
    kind="linear",
    centeredh=True,
    interp=True,
    onthefly=False,
    engine="numpy",
    dtype="float64",
):
    r"""Two dimensional Radon transform.

    Apply two dimensional Radon forward (and adjoint) transform to a
    2-dimensional array of size :math:`[n_{p_x} \times n_t]`
    (and :math:`[n_x \times n_t]`).

    In forward mode this entails to spreading the model vector
    along parametric curves (lines, parabolas, or hyperbolas depending on the
    choice of ``kind``), while stacking values in the data vector
    along the same parametric curves is performed in adjoint mode.

    Parameters
    ----------
    taxis : :obj:`np.ndarray`
        Time axis
    haxis : :obj:`np.ndarray`
        Spatial axis
    pxaxis : :obj:`np.ndarray`
        Axis of scanning variable :math:`p_x` of parametric curve
    kind : :obj:`str`, optional
        Curve to be used for stacking/spreading (``linear``, ``parabolic``,
        and ``hyperbolic`` are currently supported) or a function that takes
        :math:`(x, t_0, p_x)` as input and returns :math:`t` as output
    centeredh : :obj:`bool`, optional
        Assume centered spatial axis (``True``) or not (``False``). If ``True``
        the original ``haxis`` is ignored and a new axis is created.
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
    r2op : :obj:`pylops.LinearOperator`
        Radon operator

    Raises
    ------
    KeyError
        If ``engine`` is neither ``numpy`` nor ``numba``
    NotImplementedError
        If ``kind`` is not ``linear``, ``parabolic``, or ``hyperbolic``

    See Also
    --------
    pylops.signalprocessing.Radon3D: Three dimensional Radon transform
    pylops.Spread: Spread operator

    Notes
    -----
    The Radon2D operator applies the following linear transform in adjoint mode
    to the data after reshaping it into a 2-dimensional array of
    size :math:`[n_x \times n_t]` in adjoint mode:

    .. math::
        m(p_x, t_0) = \int{d(x, t = f(p_x, x, t))} \,\mathrm{d}x

    where :math:`f(p_x, x, t) = t_0 + p_x x` where
    :math:`p_x = \sin(\theta)/v` in linear mode,
    :math:`f(p_x, x, t) = t_0 + p_x x^2` in parabolic mode, and
    :math:`f(p_x, x, t) = \sqrt{t_0^2 + x^2 / p_x^2}` in hyperbolic mode. Note
    that internally the :math:`p_x` axis will be normalized by the ratio of the
    spatial and time axes and used alongside unitless axes. Whilst this makes
    the linear mode fully unitless, users are required to apply additional
    scalings to the :math:`p_x` axis for other relationships:

    - :math:`p_x` should be pre-multipled by :math:`d_x` for the parabolic
      relationship;
    - :math:`p_x` should be pre-multipled by :math:`(d_t/d_x)^2` for the
      hyperbolic relationship.

    As the adjoint operator can be interpreted as a repeated summation of sets
    of elements of the model vector along chosen parametric curves, the
    forward is implemented as spreading of values in the data vector along the
    same parametric curves. This operator is actually a thin wrapper around
    the :class:`pylops.Spread` operator.

    """
    # engine
    if engine not in ["numpy", "numba"]:
        raise KeyError("engine must be numpy or numba")
    if engine == "numba" and jit is None:
        engine = "numpy"
    # axes
    nt, nh, npx = taxis.size, haxis.size, pxaxis.size
    if kind == "linear":
        f = _linear if engine == "numpy" else _linear_numba
    elif kind == "parabolic":
        f = _parabolic if engine == "numpy" else _parabolic_numba
    elif kind == "hyperbolic":
        f = _hyperbolic if engine == "numpy" else _hyperbolic_numba
    elif callable(kind):
        f = kind
    else:
        raise NotImplementedError("kind must be linear, " "parabolic, or hyperbolic...")
    # make axes unitless
    dh, dt = np.abs(haxis[1] - haxis[0]), np.abs(taxis[1] - taxis[0])
    dpx = dh / dt
    pxaxis = pxaxis * dpx
    if not centeredh:
        haxisunitless = haxis // dh
    else:
        haxisunitless = np.arange(nh) - nh // 2
    dims = (npx, nt)
    dimsd = (nh, nt)

    if onthefly:
        if engine == "numba":

            @jit(nopython=True, nogil=True)
            def ontheflyfunc(x, y):
                return _indices_2d_onthefly_numba(
                    f, haxisunitless, pxaxis, x, y, nt, interp=interp
                )[1:]

        else:
            if interp:

                def ontheflyfunc(x, y):
                    return _indices_2d_onthefly(
                        f, haxisunitless, pxaxis, x, y, nt, interp=interp
                    )[1:]

            else:

                def ontheflyfunc(x, y):
                    return _indices_2d_onthefly(
                        f, haxisunitless, pxaxis, x, y, nt, interp=interp
                    )[1]

        r2op = Spread(
            dims, dimsd, fh=ontheflyfunc, interp=interp, engine=engine, dtype=dtype
        )
    else:
        if engine == "numba":
            tablefunc = _create_table_numba
        else:
            tablefunc = _create_table

        table, dtable = tablefunc(f, haxisunitless, pxaxis, nt, npx, nh, interp)
        if not interp:
            dtable = None
        r2op = Spread(
            dims,
            dimsd,
            table=table,
            dtable=dtable,
            interp=interp,
            engine=engine,
            dtype=dtype,
        )
    return r2op
