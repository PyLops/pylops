import os

import numpy as np
from numba import jit, prange

# detect whether to use parallel or not
numba_threads = int(os.getenv("NUMBA_NUM_THREADS", "1"))
parallel = True if numba_threads != 1 else False


@jit(nopython=True, parallel=parallel, nogil=True, cache=True, fastmath=True)
def _radon_inner_2d(X, Y, f, px, h, flim0, flim1, npx, nh):
    for ih in prange(nh):
        for ifr in range(flim0, flim1):
            for ipx in range(npx):
                Y[ih, ifr] += X[ipx, ifr] * np.exp(
                    -1j * 2 * np.pi * f[ifr] * px[ipx] * h[ih]
                )
    return Y


@jit(nopython=True, parallel=parallel, nogil=True, cache=True, fastmath=True)
def _aradon_inner_2d(X, Y, f, px, h, flim0, flim1, npx, nh):
    for ipx in prange(npx):
        for ifr in range(flim0, flim1):
            for ih in range(nh):
                X[ipx, ifr] += Y[ih, ifr] * np.exp(
                    1j * 2 * np.pi * f[ifr] * px[ipx] * h[ih]
                )
    return X
