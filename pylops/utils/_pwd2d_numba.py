import os

from pylops.utils.typing import NDArray

# Remap numba njit to no-ops and prange to range
# if numba is not available
try:  # pragma: no cover - executed only when numba is available
    from numba import njit, prange
except ImportError:  # pragma: no cover - executed only without numba

    def njit(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator

    def prange(*args, **kwargs):  # type: ignore
        return range(*args, **kwargs)


# Detect whether to cache or not
cache = bool(int(os.getenv("NUMBA_CACHE_PYLOPS", 0)))


@njit(fastmath=True, cache=cache)
def _B3(sigma: float) -> tuple[float, float, float]:
    """Quadratic B-spline coefficients (3 taps)."""
    b0 = (1.0 - sigma) * (2.0 - sigma) / 12.0
    b1 = (2.0 + sigma) * (2.0 - sigma) / 6.0
    b2 = (1.0 + sigma) * (2.0 + sigma) / 12.0
    return b0, b1, b2


@njit(fastmath=True, cache=cache)
def _B3d(sigma: float) -> tuple[float, float, float]:
    """Derivatives of quadratic B-spline coefficients."""
    b0 = -(2.0 - sigma) / 12.0 - (1.0 - sigma) / 12.0
    b1 = (2.0 - sigma) / 6.0 - (2.0 + sigma) / 6.0
    b2 = (2.0 + sigma) / 12.0 + (1.0 + sigma) / 12.0
    return b0, b1, b2


@njit(fastmath=True, cache=cache)
def _B5(sigma: float) -> tuple[float, float, float, float, float]:
    """Quartic B-spline coefficients (5 taps)."""
    s = sigma
    b0 = (1 - s) * (2 - s) * (3 - s) * (4 - s) / 1680.0
    b1 = (4 - s) * (2 - s) * (3 - s) * (4 + s) / 420.0
    b2 = (4 - s) * (3 - s) * (3 + s) * (4 + s) / 280.0
    b3 = (4 - s) * (2 + s) * (3 + s) * (4 + s) / 420.0
    b4 = (1 + s) * (2 + s) * (3 + s) * (4 + s) / 1680.0
    return b0, b1, b2, b3, b4


@njit(fastmath=True, cache=cache)
def _B5d(sigma: float) -> tuple[float, float, float, float, float]:
    """Derivatives of quartic B-spline coefficients."""
    s = sigma
    b0 = (
        -(
            (2 - s) * (3 - s) * (4 - s)
            + (1 - s) * (3 - s) * (4 - s)
            + (1 - s) * (2 - s) * (4 - s)
            + (1 - s) * (2 - s) * (3 - s)
        )
        / 1680.0
    )
    b1 = (
        -(
            (2 - s) * (3 - s) * (4 + s)
            + (4 - s) * (3 - s) * (4 + s)
            + (4 - s) * (2 - s) * (4 + s)
        )
        / 420.0
        + (4 - s) * (2 - s) * (3 - s) / 420.0
    )
    b2 = (
        -((3 - s) * (3 + s) * (4 + s) + (4 - s) * (3 + s) * (4 + s)) / 280.0
        + (4 - s) * (3 - s) * (4 + s) / 280.0
        + (4 - s) * (3 - s) * (3 + s) / 280.0
    )
    b3 = (
        -((2 + s) * (3 + s) * (4 + s)) / 420.0
        + (4 - s) * (3 + s) * (4 + s) / 420.0
        + (4 - s) * (2 + s) * (4 + s) / 420.0
        + (4 - s) * (2 + s) * (3 + s) / 420.0
    )
    b4 = (
        (2 + s) * (3 + s) * (4 + s)
        + (1 + s) * (3 + s) * (4 + s)
        + (1 + s) * (2 + s) * (4 + s)
        + (1 + s) * (2 + s) * (3 + s)
    ) / 1680.0
    return b0, b1, b2, b3, b4


@njit(parallel=True, fastmath=True, cache=cache)
def _conv_allpass_numba(
    din: NDArray,
    dip: NDArray,
    order: int,
    u1: NDArray,
    u2: NDArray,
) -> None:
    """Numba kernel for PWD all-pass filtering used in
    :func:`pylops.utils.signalprocessing.pwd_slope_estimate`."""
    n1, n2 = din.shape[:2]
    nw = 1 if order == 1 else 2

    # Clear derivative arrays
    u1[:] = 0.0
    u2[:] = 0.0

    for i1 in prange(nw, n1 - nw):
        for i2 in range(0, n2 - 1):
            s = dip[i1, i2]

            if order == 1:
                b0d, b1d, b2d = _B3d(s)
                b0, b1, b2 = _B3(s)

                v = din[i1 - 1, i2 + 1] - din[i1 + 1, i2]
                u1[i1, i2] += v * b0d
                u2[i1, i2] += v * b0

                v = din[i1 + 0, i2 + 1] - din[i1 + 0, i2]
                u1[i1, i2] += v * b1d
                u2[i1, i2] += v * b1

                v = din[i1 + 1, i2 + 1] - din[i1 - 1, i2]
                u1[i1, i2] += v * b2d
                u2[i1, i2] += v * b2

            else:
                c0d, c1d, c2d, c3d, c4d = _B5d(s)
                c0, c1, c2, c3, c4 = _B5(s)

                v = din[i1 - 2, i2 + 1] - din[i1 + 2, i2]
                u1[i1, i2] += v * c0d
                u2[i1, i2] += v * c0

                v = din[i1 - 1, i2 + 1] - din[i1 + 1, i2]
                u1[i1, i2] += v * c1d
                u2[i1, i2] += v * c1

                v = din[i1 + 0, i2 + 1] - din[i1 + 0, i2]
                u1[i1, i2] += v * c2d
                u2[i1, i2] += v * c2

                v = din[i1 + 1, i2 + 1] - din[i1 - 1, i2]
                u1[i1, i2] += v * c3d
                u2[i1, i2] += v * c3

                v = din[i1 + 2, i2 + 1] - din[i1 - 2, i2]
                u1[i1, i2] += v * c4d
                u2[i1, i2] += v * c4
