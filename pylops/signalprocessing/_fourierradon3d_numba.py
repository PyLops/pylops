import os

import numpy as np
from numba import jit, prange

# detect whether to use parallel or not
numba_threads = int(os.getenv("NUMBA_NUM_THREADS", "1"))
parallel = True if numba_threads != 1 else False


@jit(nopython=True, parallel=parallel, nogil=True, cache=True, fastmath=True)
def _radon_inner_3d(X, Y, f, py, px, hy, hx, flim0, flim1, npy, npx, nhy, nhx):
    for ihy in prange(nhy):
        for ihx in prange(nhx):
            for ifr in range(flim0, flim1):
                for ipy in range(npy):
                    for ipx in range(npx):
                        Y[ihy, ihx, ifr] += X[ipy, ipx, ifr] * np.exp(
                            -1j
                            * 2
                            * np.pi
                            * f[ifr]
                            * (py[ipy] * hy[ihy] + px[ipx] * hx[ihx])
                        )
    return Y


@jit(nopython=True, parallel=parallel, nogil=True, cache=True, fastmath=True)
def _aradon_inner_3d(X, Y, f, py, px, hy, hx, flim0, flim1, npy, npx, nhy, nhx):
    for ipy in prange(npy):
        for ipx in range(npx):
            for ifr in range(flim0, flim1):
                for ihy in range(nhy):
                    for ihx in range(nhx):
                        X[ipy, ipx, ifr] += Y[ihy, ihx, ifr] * np.exp(
                            1j
                            * 2
                            * np.pi
                            * f[ifr]
                            * (py[ipy] * hy[ihy] + px[ipx] * hx[ihx])
                        )
    return X
