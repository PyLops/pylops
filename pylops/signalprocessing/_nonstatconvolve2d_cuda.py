from math import floor

from numba import cuda


@cuda.jit
def _matvec_rmatvec(
    x, y, psfs, psfshape, xdims, opsfx, opsfz, dpsfx, dpsfz, npsfx, npsfz, rmatvec
):
    """Cuda kernels for NonStationaryConvolve2D operator

    Cuda implementation of matvec and rmatvec for NonStationaryConvolve2D operator. See
    :class:`pylops.signalprocessing.NonStationaryConvolve2D` for details about input parameters.

    """
    ix, iz = cuda.grid(2)

    if ix < xdims[0] and iz < xdims[1]:
        # find and interpolate psf
        ipsfx_l = int(floor((ix - opsfx) / dpsfx))
        ipsfz_t = int(floor((iz - opsfz) / dpsfz))
        dpsfx_r = (ix - opsfx) / dpsfx - ipsfx_l
        dpsfz_b = (iz - opsfz) / dpsfz - ipsfz_t
        if ipsfx_l < 0:
            ipsfx_l = ipsfx_r = 0
            dpsfx_l = dpsfx_r = 0.5
        elif ipsfx_l >= npsfx - 1:
            ipsfx_l = ipsfx_r = npsfx - 1
            dpsfx_l = dpsfx_r = 0.5
        else:
            ipsfx_r = ipsfx_l + 1
            dpsfx_l = 1.0 - dpsfx_r

        if ipsfz_t < 0:
            ipsfz_t = ipsfz_b = 0
            dpsfz_t = dpsfz_b = 0.5
        elif ipsfz_t >= npsfz - 1:
            ipsfz_t = ipsfz_b = npsfz - 1
            dpsfz_t = dpsfz_b = 0.5
        else:
            ipsfz_b = ipsfz_t + 1
            dpsfz_t = 1.0 - dpsfz_b

        psf_tl = psfs[ipsfx_l, ipsfz_t]
        psf_bl = psfs[ipsfx_l, ipsfz_b]
        psf_tr = psfs[ipsfx_r, ipsfz_t]
        psf_br = psfs[ipsfx_r, ipsfz_b]

        # find extremes of model where to apply psf (in case psf is going out of model)
        xextremes = (
            int(max(0, ix - psfshape[0] // 2)),
            int(min(ix + psfshape[0] // 2 + 1, xdims[0])),
        )
        zextremes = (
            int(max(0, iz - psfshape[1] // 2)),
            int(min(iz + psfshape[1] // 2 + 1, xdims[1])),
        )
        # find extremes of psf (in case psf is going out of model)
        psfxextremes = (
            int(max(0, -ix + psfshape[0] // 2)),
            int(min(psfshape[0], psfshape[0] // 2 + (xdims[0] - ix))),
        )
        psfzextremes = (
            int(max(0, -iz + psfshape[1] // 2)),
            int(min(psfshape[1], psfshape[1] // 2 + (xdims[1] - iz))),
        )
        # place filter in output
        for ixx, psfxx in zip(
            range(xextremes[0], xextremes[1]), range(psfxextremes[0], psfxextremes[1])
        ):
            for izz, psfzz in zip(
                range(zextremes[0], zextremes[1]),
                range(psfzextremes[0], psfzextremes[1]),
            ):
                psf = (
                    dpsfz_t * dpsfx_l * psf_tl[psfxx, psfzz]
                    + dpsfz_b * dpsfx_l * psf_bl[psfxx, psfzz]
                    + dpsfz_t * dpsfx_r * psf_tr[psfxx, psfzz]
                    + dpsfz_b * dpsfx_r * psf_br[psfxx, psfzz]
                )
                if rmatvec:
                    y[ix, iz] += psf * x[ixx, izz]
                else:
                    cuda.atomic.add(y, (ixx, izz), x[ix, iz] * psf)


def _matvec_rmatvec_call(
    x,
    y,
    psfs,
    psfshape,
    xdims,
    opsfx,
    opsfz,
    dpsfx,
    dpsfz,
    npsfx,
    npsfz,
    rmatvec=False,
    num_blocks=(32, 32),
    num_threads_per_blocks=(32, 32),
):
    """Caller for NonStationaryConvolve2D operator

    Caller for cuda implementation of matvec and rmatvec for NonStationaryConvolve2D operato, with same signature
    as numpy/numba counterparts. See :class:`pylops.signalprocessing.NonStationaryConvolve2D` for details about
     input parameters.

    """
    _matvec_rmatvec[num_blocks, num_threads_per_blocks](
        x, y, psfs, psfshape, xdims, opsfx, opsfz, dpsfx, dpsfz, npsfx, npsfz, rmatvec
    )
    return y
