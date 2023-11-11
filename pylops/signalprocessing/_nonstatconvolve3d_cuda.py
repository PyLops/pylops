from math import floor

from numba import cuda


@cuda.jit(max_registers=40)
def _matvec_rmatvec(
    x, y, hs, hshape, xdims, ohx, ohy, ohz, dhx, dhy, dhz, nhx, nhy, nhz, rmatvec
):
    """Cuda kernels for NonStationaryConvolve3D operator

    Cuda implementation of matvec and rmatvec for NonStationaryConvolve3D operator. See
    :class:`pylops.signalprocessing.NonStationaryConvolve3D` for details about input parameters.

    """
    ix, iy, iz = cuda.grid(3)

    if ix < xdims[0] and iy < xdims[1] and iz < xdims[2]:

        # find closest filters and interpolate h
        ihx_l = int(floor((ix - ohx) / dhx))  # id number of left for hs_arr
        ihy_b = int(floor((iy - ohy) / dhy))  # id number of back for hs_arr
        ihz_t = int(floor((iz - ohz) / dhz))  # id number of top  for hs_arr

        dhx_r = (ix - ohx) / dhx - ihx_l  # weight for right psfs, left 1-dhx_r
        dhy_f = (iy - ohy) / dhy - ihy_b  # weight for front psfs, back 1-dhy_f
        dhz_d = (iz - ohz) / dhz - ihz_t  # weight for down psfs,  top 1-dhz_d

        if ihx_l < 0:
            ihx_l = ihx_r = 0
            dhx_l = dhx_r = 0.5
        elif ihx_l >= nhx - 1:
            ihx_l = ihx_r = nhx - 1
            dhx_l = dhx_r = 0.5
        else:
            ihx_r = ihx_l + 1
            dhx_l = 1.0 - dhx_r

        if ihy_b < 0:
            ihy_b = ihy_f = 0
            dhy_b = dhy_f = 0.5
        elif ihy_b >= nhy - 1:
            ihy_b = ihy_f = nhy - 1
            dhy_b = dhy_f = 0.5
        else:
            ihy_f = ihy_b + 1
            dhy_b = 1.0 - dhy_f

        if ihz_t < 0:
            ihz_t = ihz_d = 0
            dhz_t = dhz_d = 0.5
        elif ihz_t >= nhz - 1:
            ihz_t = ihz_d = nhz - 1
            dhz_t = dhz_d = 0.5
        else:
            ihz_d = ihz_t + 1
            dhz_t = 1.0 - dhz_d

        h_lbt = hs[ihx_l, ihy_b, ihz_t]
        h_lbd = hs[ihx_l, ihy_b, ihz_d]
        h_lft = hs[ihx_l, ihy_f, ihz_t]
        h_lfd = hs[ihx_l, ihy_f, ihz_d]

        h_rbt = hs[ihx_r, ihy_b, ihz_t]
        h_rbd = hs[ihx_r, ihy_b, ihz_d]
        h_rft = hs[ihx_r, ihy_f, ihz_t]
        h_rfd = hs[ihx_r, ihy_f, ihz_d]

        # find extremes of model where to apply h (in case h is going out of model)
        xextremes = (
            max(0, ix - hshape[0] // 2),
            min(ix + hshape[0] // 2 + 1, xdims[0]),
        )
        yextremes = (
            max(0, iy - hshape[1] // 2),
            min(iy + hshape[1] // 2 + 1, xdims[1]),
        )
        zextremes = (
            max(0, iz - hshape[2] // 2),
            min(iz + hshape[2] // 2 + 1, xdims[2]),
        )

        # find extremes of h (in case h is going out of model)
        hxextremes = (
            max(0, -ix + hshape[0] // 2),
            min(hshape[0], hshape[0] // 2 + (xdims[0] - ix)),
        )
        hyextremes = (
            max(0, -iy + hshape[1] // 2),
            min(hshape[1], hshape[1] // 2 + (xdims[1] - iy)),
        )
        hzextremes = (
            max(0, -iz + hshape[2] // 2),
            min(hshape[2], hshape[2] // 2 + (xdims[2] - iz)),
        )

        # place filter in output
        for ixx, hxx in zip(
            range(xextremes[0], xextremes[1]), range(hxextremes[0], hxextremes[1])
        ):
            for iyy, hyy in zip(
                range(yextremes[0], yextremes[1]), range(hyextremes[0], hyextremes[1])
            ):
                for izz, hzz in zip(
                    range(zextremes[0], zextremes[1]),
                    range(hzextremes[0], hzextremes[1]),
                ):
                    h = (
                        dhx_l * dhy_b * dhz_t * h_lbt[hxx, hyy, hzz]
                        + dhx_l * dhy_b * dhz_d * h_lbd[hxx, hyy, hzz]
                        + dhx_l * dhy_f * dhz_t * h_lft[hxx, hyy, hzz]
                        + dhx_l * dhy_f * dhz_d * h_lfd[hxx, hyy, hzz]
                        + dhx_r * dhy_b * dhz_t * h_rbt[hxx, hyy, hzz]
                        + dhx_r * dhy_b * dhz_d * h_rbd[hxx, hyy, hzz]
                        + dhx_r * dhy_f * dhz_t * h_rft[hxx, hyy, hzz]
                        + dhx_r * dhy_f * dhz_d * h_rfd[hxx, hyy, hzz]
                    )

                    if rmatvec:
                        y[ix, iy, iz] += h * x[ixx, iyy, izz]
                    else:
                        cuda.atomic.add(y, (ixx, iyy, izz), x[ix, iy, iz] * h)


def _matvec_rmatvec_call(
    x,
    y,
    hs,
    hshape,
    xdims,
    ohx,
    ohy,
    ohz,
    dhx,
    dhy,
    dhz,
    nhx,
    nhy,
    nhz,
    rmatvec=False,
    num_blocks=(1, 32, 16),
    num_threads_per_blocks=(24, 24, 24),
):
    """Caller for NonStationaryConvolve3D operator

    Caller for cuda implementation of matvec and rmatvec for NonStationaryConvolve3D operator, with same signature
    as numpy/numba counterparts. See :class:`pylops.signalprocessing.NonStationaryConvolve3D` for details about
     input parameters.

    """
    _matvec_rmatvec[num_blocks, num_threads_per_blocks](
        x, y, hs, hshape, xdims, ohx, ohy, ohz, dhx, dhy, dhz, nhx, nhy, nhz, rmatvec
    )
    return y
