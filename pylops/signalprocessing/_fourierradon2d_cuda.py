from cmath import exp
from math import pi

from numba import cuda


@cuda.jit()
def _radon_inner_2d_kernel(x, y, f, px, h, flim0, flim1, npx, nh):
    """Cuda kernels for FourierRadon2D operator

    Cuda implementation of the on-the-fly kernel creation and application for the
    FourierRadon2D operator. See :class:`pylops.signalprocessing.FourierRadon2D`
    for details about input parameters.

    """
    ih, ifr = cuda.grid(2)
    if ih < nh and ifr >= flim0 and ifr <= flim1:
        for ipx in range(npx):
            y[ih, ifr] += x[ipx, ifr] * exp(-1j * 2 * pi * f[ifr] * px[ipx] * h[ih])


@cuda.jit()
def _aradon_inner_2d_kernel(x, y, f, px, h, flim0, flim1, npx, nh):
    """Cuda kernels for FourierRadon2D operator

    Cuda implementation of the on-the-fly kernel creation and application for the
    FourierRadon2D operator. See :class:`pylops.signalprocessing.FourierRadon2D`
    for details about input parameters.

    """
    ipx, ifr = cuda.grid(2)
    if ipx < npx and ifr >= flim0 and ifr <= flim1:
        for ih in range(nh):
            x[ipx, ifr] += y[ih, ifr] * exp(1j * 2 * pi * f[ifr] * px[ipx] * h[ih])


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
