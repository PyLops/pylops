import os
import numpy as np
from numba import jit

# detect whether to use parallel or not
numba_threads = int(os.getenv('NUMBA_NUM_THREADS', '1'))
parallel = True if numba_threads != 1 else False


@jit(nopython=True)
def _linear_numba(y, x, t, py, px):
    return t + px*x + py*y

@jit(nopython=True)
def _parabolic_numba(y, x, t, py, px):
    return t + px*x**2 + py*y**2

@jit(nopython=True)
def _hyperbolic_numba(y, x, t, py, px):
    return np.sqrt(t**2 + (x/px)**2 + (y/py)**2)

@jit(nopython=True, parallel=parallel, nogil=True)
def _indices_3d_numba(f, y, x, py, px, t, nt, interp=True):
    """Compute time and space indices of parametric line in ``f`` function
    using numba. Refer to ``_indices_3d`` for full documentation.
    """
    tdecscan = f(y, x, t, py, px)
    if not interp:
        sscan = (tdecscan >= 0) & (tdecscan < nt)
    else:
        sscan = (tdecscan >= 0) & (tdecscan < nt - 1)
    tscanfs = tdecscan[sscan]
    tscan = np.zeros(len(tscanfs))
    dtscan = np.zeros(len(tscanfs))
    for it, tscanf in enumerate(tscanfs):
        tscan[it] = int(tscanf)
        if interp:
            dtscan[it] = tscanf - tscan[it]
    return sscan, tscan, dtscan

@jit(nopython=True, parallel=parallel, nogil=True)
def _indices_3d_onthefly_numba(f, y, x, py, px, ip, t, nt, interp=True):
    """Wrapper around _indices_3d to allow on-the-fly computation of
    parametric curves using numba
    """
    tscan = np.full(len(y), np.nan, dtype=np.float32)
    if interp:
        dtscan = np.full(len(y), np.nan)
    else:
        dtscan = None
    sscan, tscan1, dtscan1 = \
        _indices_3d_numba(f, y, x, py[ip], px[ip], t, nt, interp=interp)
    tscan[sscan] = tscan1
    if interp:
        dtscan[sscan] = dtscan1
    return sscan, tscan, dtscan

@jit(nopython=True, parallel=parallel, nogil=True)
def _create_table_numba(f, y, x, pyaxis, pxaxis, nt, npy, npx, ny, nx, interp):
    """Create look up table using numba
    """
    table = np.full((npx * npy, nt, ny * nx), np.nan, dtype=np.float32)
    dtable = np.full((npx * npy, nt, ny * nx), np.nan)
    for ip in range(len(pyaxis)):
        py = pyaxis[ip]
        px = pxaxis[ip]
        for it in range(nt):
            sscans, tscan, dtscan = _indices_3d_numba(f, y, x,
                                                     py, px,
                                                     it, nt,
                                                     interp=interp)
            itscan = 0
            for isscan, sscan in enumerate(sscans):
                if sscan:
                    table[ip, it, isscan] = tscan[itscan]
                    if interp:
                        dtable[ip, it, isscan] = dtscan[itscan]
                    itscan += 1
    return table, dtable
