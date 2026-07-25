import os
from cmath import exp
from math import pi

from numba import jit, prange

# Detect whether to use parallel and cache or not
numba_threads = int(os.getenv("NUMBA_NUM_THREADS", "1"))
parallel = True if numba_threads != 1 else False
cache = bool(int(os.getenv("NUMBA_CACHE_PYLOPS", 0)))


@jit(nopython=True, parallel=parallel, nogil=True, cache=cache, fastmath=True)
def _radon_inner_2d(X, Y, f, px, h, flim0, flim1, npx, nh):
    for ih in prange(nh):
        for ifr in range(flim0, flim1):
            for ipx in range(npx):
                Y[ih, ifr] += X[ipx, ifr] * exp(-1j * 2 * pi * f[ifr] * px[ipx] * h[ih])


@jit(nopython=True, parallel=parallel, nogil=True, cache=cache, fastmath=True)
def _aradon_inner_2d(X, Y, f, px, h, flim0, flim1, npx, nh):
    for ipx in prange(npx):
        for ifr in range(flim0, flim1):
            for ih in range(nh):
                X[ipx, ifr] += Y[ih, ifr] * exp(1j * 2 * pi * f[ifr] * px[ipx] * h[ih])
