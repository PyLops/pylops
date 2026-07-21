import os
from math import floor

from pylops.utils.typing import NDArray

# Remap numba njit to no-ops and prange to range
# if numba is not available
try:  # pragma: no cover - executed only when numba is available
    from numba import njit, prange
except ImportError:  # pragma: no cover - executed only in no-numba builds

    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator

    def prange(*args, **kwargs):  # type: ignore
        return range(*args, **kwargs)


# Detect whether to cache or not
cache = bool(int(os.getenv("NUMBA_CACHE_PYLOPS", 0)))


@njit(cache=cache, fastmath=True, parallel=False)
def _spray_forward_numba(
    m: NDArray, sigma: NDArray, radius: int, alpha: float, out: NDArray
) -> None:
    """Forward spraying kernel used in
    :class:`pylops.signalprocessing.PWSprayer2D`.
    """
    nz, nx = m.shape
    ftype = sigma.dtype.type
    alpha = ftype(alpha)
    for z0 in prange(nz):
        for x0 in range(nx):
            v0 = m[z0, x0]
            if v0 == m.dtype.type(0.0):
                continue

            out[z0, x0] += v0

            amp = v0
            z = float(z0)
            for k in range(1, radius + 1):
                x = x0 + k
                if x >= nx:
                    break
                s = sigma[z0, min(x - 1, nx - 1)]
                z += s
                amp *= alpha
                zi = floor(z)
                t = z - zi
                zi0 = 0 if zi < 0 else (nz - 1 if zi >= nz else zi)
                zi1 = 0 if zi + 1 < 0 else (nz - 1 if zi + 1 >= nz else zi + 1)
                out[zi0, x] += (1 - t) * amp
                out[zi1, x] += t * amp

            amp = v0
            z = float(z0)
            for k in range(1, radius + 1):
                x = x0 - k
                if x < 0:
                    break
                s = sigma[z0, x]
                z -= s
                amp *= alpha
                zi = floor(z)
                t = z - zi
                zi0 = 0 if zi < 0 else (nz - 1 if zi >= nz else zi)
                zi1 = 0 if zi + 1 < 0 else (nz - 1 if zi + 1 >= nz else zi + 1)
                out[zi0, x] += (1 - t) * amp
                out[zi1, x] += t * amp


@njit(cache=cache, fastmath=True, parallel=True)
def _spray_adjoint_numba(
    d: NDArray, sigma: NDArray, radius: int, alpha: float, out: NDArray
) -> None:
    """Adjoint spraying kernel used in
    :class:`pylops.signalprocessing.PWSprayer2D`.
    """
    nz, nx = d.shape
    ftype = sigma.dtype.type
    alpha = ftype(alpha)
    for z0 in prange(nz):
        for x0 in range(nx):
            acc = d[z0, x0]

            amp = ftype(1.0)
            z = ftype(z0)
            for k in range(1, radius + 1):
                x = x0 + k
                if x >= nx:
                    break
                s = sigma[z0, min(x - 1, nx - 1)]
                z += s
                amp *= alpha
                zi = floor(z)
                t = z - zi
                zi0 = 0 if zi < 0 else (nz - 1 if zi >= nz else zi)
                zi1 = 0 if zi + 1 < 0 else (nz - 1 if zi + 1 >= nz else zi + 1)
                acc += (1 - t) * amp * d[zi0, x]
                acc += t * amp * d[zi1, x]

            amp = ftype(1.0)
            z = ftype(z0)
            for k in range(1, radius + 1):
                x = x0 - k
                if x < 0:
                    break
                s = sigma[z0, x]
                z -= s
                amp *= alpha
                zi = floor(z)
                t = z - zi
                zi0 = 0 if zi < 0 else (nz - 1 if zi >= nz else zi)
                zi1 = 0 if zi + 1 < 0 else (nz - 1 if zi + 1 >= nz else zi + 1)
                acc += (1 - t) * amp * d[zi0, x]
                acc += t * amp * d[zi1, x]

            out[z0, x0] += acc
