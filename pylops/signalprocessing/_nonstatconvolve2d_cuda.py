from math import floor

from numba import cuda


@cuda.jit(max_registers=40)
def _matvec_rmatvec(x, y, hs, hshape, xdims, ohx, ohz, dhx, dhz, nhx, nhz, rmatvec):
    """Cuda kernels for NonStationaryConvolve2D operator

    Cuda implementation of matvec and rmatvec for NonStationaryConvolve2D operator. See
    :class:`pylops.signalprocessing.NonStationaryConvolve2D` for details about input parameters.

    """
    ix, iz = cuda.grid(2)

    if ix < xdims[0] and iz < xdims[1]:
        # find and interpolate h
        ihx_l = int(floor((ix - ohx) / dhx))
        ihz_t = int(floor((iz - ohz) / dhz))
        dhx_r = (ix - ohx) / dhx - ihx_l
        dhz_b = (iz - ohz) / dhz - ihz_t
        if ihx_l < 0:
            ihx_l = ihx_r = 0
            dhx_l = dhx_r = 0.5
        elif ihx_l >= nhx - 1:
            ihx_l = ihx_r = nhx - 1
            dhx_l = dhx_r = 0.5
        else:
            ihx_r = ihx_l + 1
            dhx_l = 1.0 - dhx_r

        if ihz_t < 0:
            ihz_t = ihz_b = 0
            dhz_t = dhz_b = 0.5
        elif ihz_t >= nhz - 1:
            ihz_t = ihz_b = nhz - 1
            dhz_t = dhz_b = 0.5
        else:
            ihz_b = ihz_t + 1
            dhz_t = 1.0 - dhz_b

        h_tl = hs[ihx_l, ihz_t]
        h_bl = hs[ihx_l, ihz_b]
        h_tr = hs[ihx_r, ihz_t]
        h_br = hs[ihx_r, ihz_b]

        # find extremes of model where to apply h (in case h is going out of model)
        xextremes = (
            int(max(0, ix - hshape[0] // 2)),
            int(min(ix + hshape[0] // 2 + 1, xdims[0])),
        )
        zextremes = (
            int(max(0, iz - hshape[1] // 2)),
            int(min(iz + hshape[1] // 2 + 1, xdims[1])),
        )
        # find extremes of h (in case h is going out of model)
        hxextremes = (
            int(max(0, -ix + hshape[0] // 2)),
            int(min(hshape[0], hshape[0] // 2 + (xdims[0] - ix))),
        )
        hzextremes = (
            int(max(0, -iz + hshape[1] // 2)),
            int(min(hshape[1], hshape[1] // 2 + (xdims[1] - iz))),
        )
        # place filter in output
        for ixx, hxx in zip(
            range(xextremes[0], xextremes[1]), range(hxextremes[0], hxextremes[1])
        ):
            for izz, hzz in zip(
                range(zextremes[0], zextremes[1]),
                range(hzextremes[0], hzextremes[1]),
            ):
                h = (
                    dhz_t * dhx_l * h_tl[hxx, hzz]
                    + dhz_b * dhx_l * h_bl[hxx, hzz]
                    + dhz_t * dhx_r * h_tr[hxx, hzz]
                    + dhz_b * dhx_r * h_br[hxx, hzz]
                )
                if rmatvec:
                    y[ix, iz] += h * x[ixx, izz]
                else:
                    cuda.atomic.add(y, (ixx, izz), x[ix, iz] * h)


def _matvec_rmatvec_call(
    x,
    y,
    hs,
    hshape,
    xdims,
    ohx,
    ohz,
    dhx,
    dhz,
    nhx,
    nhz,
    rmatvec=False,
    num_blocks=(32, 32),
    num_threads_per_blocks=(32, 32),
):
    """Caller for NonStationaryConvolve2D operator

    Caller for cuda implementation of matvec and rmatvec for NonStationaryConvolve2D operator, with same signature
    as numpy/numba counterparts. See :class:`pylops.signalprocessing.NonStationaryConvolve2D` for details about
    input parameters.

    """
    _matvec_rmatvec[num_blocks, num_threads_per_blocks](
        x, y, hs, hshape, xdims, ohx, ohz, dhx, dhz, nhx, nhz, rmatvec
    )
    return y
