from math import pi

import cupy as cp
from numba import cuda

TWO_PI_MINUS = cp.float32(-2.0 * pi)
TWO_PI_PLUS = cp.float32(2.0 * pi)
IMG = cp.complex64(1j)


@cuda.jit
def _radon_inner_2d_kernel(x, y, f, px, h, flim0, flim1, npx, nh):
    """Cuda kernels for FourierRadon2D operator

    Cuda implementation of the on-the-fly kernel creation and application for the
    FourierRadon2D operator. See :class:`pylops.signalprocessing.FourierRadon2D`
    for details about input parameters.

    """
    ih, ifr = cuda.grid(2)
    if ih < nh and ifr >= flim0 and ifr <= flim1:
        for ipx in range(npx):
            # slow computation of exp(1j * x)
            # y[ih, ifr] += x[ipx, ifr] * exp(TWO_PI_MINUS * f[ifr] * px[ipx] * h[ih])
            # fast computation of exp(1j * x) - see https://stackoverflow.com/questions/9860711/cucomplex-h-and-exp/9863048#9863048
            s, c = cuda.libdevice.sincosf(TWO_PI_MINUS * f[ifr] * px[ipx] * h[ih])
            y[ih, ifr] += x[ipx, ifr] * (c + IMG * s)


@cuda.jit
def _aradon_inner_2d_kernel(x, y, f, px, h, flim0, flim1, npx, nh):
    """Cuda kernels for FourierRadon2D operator

    Cuda implementation of the on-the-fly kernel creation and application for the
    FourierRadon2D operator. See :class:`pylops.signalprocessing.FourierRadon2D`
    for details about input parameters.

    """
    ipx, ifr = cuda.grid(2)
    if ipx < npx and ifr >= flim0 and ifr <= flim1:
        for ih in range(nh):
            # slow computation of exp(1j * x)
            # x[ipx, ifr] += y[ih, ifr] * exp(TWO_PI_I_PLUS * f[ifr] * px[ipx] * h[ih])
            # fast computation of exp(1j * x) - see https://stackoverflow.com/questions/9860711/cucomplex-h-and-exp/9863048#9863048
            s, c = cuda.libdevice.sincosf(TWO_PI_PLUS * f[ifr] * px[ipx] * h[ih])
            x[ipx, ifr] += y[ih, ifr] * (c + IMG * s)


def _radon_inner_2d_cuda(
    x,
    y,
    f,
    px,
    h,
    flim0,
    flim1,
    npx,
    nh,
    num_blocks=(32, 32),
    num_threads_per_blocks=(32, 32),
):
    """Caller for FourierRadon2D operator

    Caller for cuda implementation of matvec kernel for FourierRadon2D operator.
    See :class:`pylops.signalprocessing.FourierRadon2D` for details about
    input parameters.

    """
    _radon_inner_2d_kernel[num_blocks, num_threads_per_blocks](
        x, y, f, px, h, flim0, flim1, npx, nh
    )
    return y


def _aradon_inner_2d_cuda(
    x,
    y,
    f,
    px,
    h,
    flim0,
    flim1,
    npx,
    nh,
    num_blocks=(32, 32),
    num_threads_per_blocks=(32, 32),
):
    """Caller for FourierRadon2D operator

    Caller for cuda implementation of rmatvec kernel for FourierRadon2D operator.
    See :class:`pylops.signalprocessing.FourierRadon2D` for details about
    input parameters.

    """
    _aradon_inner_2d_kernel[num_blocks, num_threads_per_blocks](
        x, y, f, px, h, flim0, flim1, npx, nh
    )
    return x
