from math import pi

import cupy as cp
from numba import cuda

TWO_PI_MINUS = cp.float32(-2.0 * pi)
TWO_PI_PLUS = cp.float32(2.0 * pi)
IMG = cp.complex64(1j)


@cuda.jit
def _radon_inner_3d_kernel(x, y, f, py, px, hy, hx, flim0, flim1, npy, npx, nhy, nhx):
    """Cuda kernels for FourierRadon3D operator

    Cuda implementation of the on-the-fly kernel creation and application for the
    FourierRadon3D operator. See :class:`pylops.signalprocessing.FourierRadon3D`
    for details about input parameters.

    """
    ihy, ihx, ifr = cuda.grid(3)
    if ihy < nhy and ihx < nhx and ifr >= flim0 and ifr <= flim1:
        for ipy in range(npy):
            for ipx in range(npx):
                # slow computation of exp(1j * x)
                # y[ihy, ihx, ifr] += x[ipy, ipx, ifr] * exp(
                #     TWO_PI_MINUS * f[ifr] * (py[ipy] * hy[ihy] + px[ipx] * hx[ihx])
                # )
                # fast computation of exp(1j * x) - see https://stackoverflow.com/questions/9860711/cucomplex-h-and-exp/9863048#9863048
                s, c = cuda.libdevice.sincosf(
                    TWO_PI_MINUS * f[ifr] * (py[ipy] * hy[ihy] + px[ipx] * hx[ihx])
                )
                y[ihy, ihx, ifr] += x[ipy, ipx, ifr] * (c + IMG * s)


@cuda.jit
def _aradon_inner_3d_kernel(x, y, f, py, px, hy, hx, flim0, flim1, npy, npx, nhy, nhx):
    """Cuda kernels for FourierRadon3D operator

    Cuda implementation of the on-the-fly kernel creation and application for the
    FourierRadon3D operator. See :class:`pylops.signalprocessing.FourierRadon3D`
    for details about input parameters.

    """
    ipy, ipx, ifr = cuda.grid(3)
    if ipy < npy and ipx < npx and ifr >= flim0 and ifr <= flim1:
        for ihy in range(nhy):
            for ihx in range(nhx):
                # slow computation of exp(1j * x)
                # x[ipy, ipx, ifr] += y[ihy, ihx, ifr] * exp(
                #     TWO_PI_I_PLUS * f[ifr] * (py[ipy] * hy[ihy] + px[ipx] * hx[ihx])
                # )
                s, c = cuda.libdevice.sincosf(
                    TWO_PI_PLUS * f[ifr] * (py[ipy] * hy[ihy] + px[ipx] * hx[ihx])
                )
                x[ipy, ipx, ifr] += y[ihy, ihx, ifr] * (c + IMG * s)


def _radon_inner_3d_cuda(
    x,
    y,
    f,
    py,
    px,
    hy,
    hx,
    flim0,
    flim1,
    npy,
    npx,
    nhy,
    nhx,
    num_blocks=(8, 8),
    num_threads_per_blocks=(8, 8),
):
    """Caller for FourierRadon2D operator

    Caller for cuda implementation of matvec kernel for FourierRadon3D operator.
    See :class:`pylops.signalprocessing.FourierRadon3D` for details about
    input parameters.

    """
    _radon_inner_3d_kernel[num_blocks, num_threads_per_blocks](
        x, y, f, py, px, hy, hx, flim0, flim1, npy, npx, nhy, nhx
    )
    return y


def _aradon_inner_3d_cuda(
    x,
    y,
    f,
    py,
    px,
    hy,
    hx,
    flim0,
    flim1,
    npy,
    npx,
    nhy,
    nhx,
    num_blocks=(8, 8),
    num_threads_per_blocks=(8, 8),
):
    """Caller for FourierRadon3D operator

    Caller for cuda implementation of rmatvec kernel for FourierRadon3D operator.
    See :class:`pylops.signalprocessing.FourierRadon3D` for details about
    input parameters.

    """
    _aradon_inner_3d_kernel[num_blocks, num_threads_per_blocks](
        x, y, f, py, px, hy, hx, flim0, flim1, npy, npx, nhy, nhx
    )
    return x
