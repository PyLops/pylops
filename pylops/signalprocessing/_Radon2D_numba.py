import os
import numpy as np
from numba import jit

# detect whether to use parallel or not
numba_threads = int(os.getenv('NUMBA_NUM_THREADS', '1'))
parallel = True if numba_threads != 1 else False


@jit(nopython=True)
def _linear_numba(x, t, px):
    return t + px * x

@jit(nopython=True)
def _parabolic_numba(x, t, px):
    return t + px*x**2

@jit(nopython=True)
def _hyperbolic_numba(x, t, px):
    return np.sqrt(t**2 + (x/px)**2)

@jit(nopython=True, nogil=True)
def _indices_2d_numba(f, x, px, t, nt, interp=True):
    """Compute time and space indices of parametric line in ``f`` function
    using numba. Refer to ``_indices_2d`` for full documentation.

    """
    tdecscan = f(x, t, px)
    if not interp:
        xscan = (tdecscan >= 0) & (tdecscan < nt)
    else:
        xscan = (tdecscan >= 0) & (tdecscan < nt - 1)
    tscanfs = tdecscan[xscan]
    tscan = np.zeros(len(tscanfs))
    dtscan = np.zeros(len(tscanfs))
    for it, tscanf in enumerate(tscanfs):
        tscan[it] = int(tscanf)
        if interp:
            dtscan[it] = tscanf - tscan[it]
    return xscan, tscan, dtscan

@jit(nopython=True, parallel=parallel, nogil=True)
def _indices_2d_onthefly_numba(f, x, px, ip, t, nt, interp=True):
    """Wrapper around _indices_2d to allow on-the-fly computation of
    parametric curves using numba
    """
    tscan = np.full(len(x), np.nan, dtype=np.float32)
    if interp:
        dtscan = np.full(len(x), np.nan)
    else:
        dtscan = None
    xscan, tscan1, dtscan1 = \
        _indices_2d_numba(f, x, px[ip], t, nt, interp=interp)
    tscan[xscan] = tscan1
    if interp:
        dtscan[xscan] = dtscan1
    return xscan, tscan, dtscan

@jit(nopython=True, parallel=parallel, nogil=True)
def _create_table_numba(f, x, pxaxis, nt, npx, nx, interp):
    """Create look up table using numba
    """
    table = np.full((npx, nt, nx), np.nan, dtype=np.float32)
    dtable = np.full((npx, nt, nx), np.nan)
    for ipx in range(npx):
        px = pxaxis[ipx]
        for it in range(nt):
            xscans, tscan, dtscan = _indices_2d_numba(f, x, px,
                                                      it, nt,
                                                      interp=interp)
            itscan = 0
            for ixscan, xscan in enumerate(xscans):
                if xscan:
                    table[ipx, it, ixscan] = tscan[itscan]
                    if interp:
                        dtable[ipx, it, ixscan] = dtscan[itscan]
                    itscan += 1
    return table, dtable
