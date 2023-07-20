__all__ = ["NonStationaryConvolve3D"]

import os
from typing import Tuple, Union

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import get_array_module
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

jit_message = deps.numba_import("the nonstatconvolve3d module")

if jit_message is None:
    from numba import jit, prange

    from ._nonstatconvolve2d_cuda import (
        _matvec_rmatvec_call as _matvec_rmatvec_cuda_call,
    )

    # detect whether to use parallel or not
    numba_threads = int(os.getenv("NUMBA_NUM_THREADS", "1"))
    parallel = True if numba_threads != 1 else False
else:
    prange = range


class NonStationaryConvolve3D(LinearOperator):
    r"""3D non-stationary convolution operator.

    Apply non-stationary three-dimensional convolution. A varying compact filter
    is provided on a coarser grid and on-the-fly interpolation is applied
    in forward and adjoint modes. Both input and output have size :math:`n_x \times n_y \times n_z`.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension (which we refer to as :math:`n_x \times n_y \times n_z`).
    hs : :obj:`numpy.ndarray`
        Bank of 3d compact filters of size
        :math:`n_{\text{filts},x} \times n_{\text{filts},y} \times
        n_{\text{filts},z} \times n_{h,x} \times n_{h,y} \times n_{h,z}`.
        Filters must have odd number of samples and are assumed to be
        centered in the middle of the filter support.
    ihx : :obj:`tuple`
        Indices of the x locations of the filters ``hs`` in the model (and data). Note
        that the filters must be regularly sampled, i.e. :math:`dh_x=\text{diff}(ihx)=\text{const.}`
    ihy : :obj:`tuple`
        Indices of the y locations of the filters ``hs`` in the model (and data). Note
        that the filters must be regularly sampled, i.e. :math:`dh_y=\text{diff}(ihy)=\text{const.}`
    ihz : :obj:`tuple`
        Indices of the z locations of the filters ``hs`` in the model (and data). Note
        that the filters must be regularly sampled, i.e. :math:`dh_z=\text{diff}(ihz)=\text{const.}`
    engine : :obj:`str`, optional
        Engine used for spread computation (``numpy``, ``numba``, or ``cuda``)
    num_threads_per_blocks : :obj:`tuple`, optional
        Number of threads in each block (only when ``engine=cuda``)
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Raises
    ------
    ValueError
        If filters ``hs`` have even size
    ValueError
        If ``ihx``, ``ihy`` or ``ihz`` is not regularly sampled
    NotImplementedError
        If ``engine`` is neither ``numpy``, ``fftw``, nor ``scipy``.

    Notes
    -----
    See :class:`pylops.signalprocessing.NonStationaryConvolve2D`.

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        hs: NDArray,
        ihx: InputDimsLike,
        ihy: InputDimsLike,
        ihz: InputDimsLike,
        engine: str = "numpy",
        num_threads_per_blocks: Tuple[int, int, int] = (2, 16, 16),
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        if engine not in ["numpy", "numba", "cuda"]:
            raise NotImplementedError("engine must be numpy or numba or cuda")
        if hs.shape[3] % 2 == 0 or hs.shape[4] % 2 == 0 or hs.shape[5] % 2 == 0:
            raise ValueError("filters hs must have odd length")
        if (
            len(np.unique(np.diff(ihx))) > 1
            or len(np.unique(np.diff(ihy))) > 1
            or len(np.unique(np.diff(ihz))) > 1
        ):
            raise ValueError(
                "the indices of filters 'ih' are must be regularly sampled"
            )
        if (
            min(ihx) < 0
            or min(ihy) < 0
            or min(ihz) < 0
            or max(ihx) >= dims[0]
            or max(ihy) >= dims[1]
            or max(ihz) >= dims[2]
        ):
            raise ValueError(
                "the indices of filters 'ih' must be larger than 0 and smaller than `dims`"
            )
        self.hs = hs
        self.hshape = hs.shape[3:]
        self.ohx, self.dhx, self.nhx = ihx[0], ihx[1] - ihx[0], len(ihx)
        self.ohy, self.dhy, self.nhy = ihy[0], ihy[1] - ihy[0], len(ihy)
        self.ohz, self.dhz, self.nhz = ihz[0], ihz[1] - ihz[0], len(ihz)
        self.ehx, self.ehx, self.ehz = ihx[-1], ihy[-1], ihz[-1]
        self.dims = _value_or_sized_to_tuple(dims)
        self.engine = engine
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

        # create additional input parameters for engine=cuda
        self.kwargs_cuda = {}
        if engine == "cuda":
            self.kwargs_cuda["num_threads_per_blocks"] = num_threads_per_blocks
            num_blocks_x = (
                self.dims[0] + num_threads_per_blocks[0] - 1
            ) // num_threads_per_blocks[0]
            num_blocks_y = (
                self.dims[1] + num_threads_per_blocks[1] - 1
            ) // num_threads_per_blocks[1]
            num_blocks_z = (
                self.dims[2] + num_threads_per_blocks[2] - 1
            ) // num_threads_per_blocks[2]
            self.kwargs_cuda["num_blocks"] = (num_blocks_x, num_blocks_y, num_blocks_z)
        self._register_multiplications(engine)

    def _register_multiplications(self, engine: str) -> None:
        if engine == "numba":
            numba_opts = dict(nopython=True, fastmath=True, nogil=True, parallel=True)
            self._mvrmv = jit(**numba_opts)(self._matvec_rmatvec)
        elif engine == "cuda":
            self._mvrmv = _matvec_rmatvec_cuda_call
        else:
            self._mvrmv = self._matvec_rmatvec

    @staticmethod
    def _matvec_rmatvec(
        x: NDArray,
        y: NDArray,
        hs: NDArray,
        hshape: Tuple[int, int, int],
        xdims: Tuple[int, int, int],
        ohx: float,
        ohy: float,
        ohz: float,
        dhx: float,
        dhy: float,
        dhz: float,
        nhx: int,
        nhy: int,
        nhz: int,
        rmatvec: bool = False,
    ) -> NDArray:
        for ix in prange(xdims[0]):
            for iy in range(xdims[1]):
                for iz in range(xdims[2]):
                    # find closest filters and interpolate h
                    ihx_l = int(
                        np.floor((ix - ohx) / dhx)
                    )  # id number of left for hs_arr
                    ihy_b = int(
                        np.floor((iy - ohy) / dhy)
                    )  # id number of back for hs_arr
                    ihz_t = int(
                        np.floor((iz - ohz) / dhz)
                    )  # id number of top  for hs_arr

                    dhx_r = (
                        ix - ohx
                    ) / dhx - ihx_l  # weight for right psfs, left 1-ihz_t
                    dhy_f = (
                        iy - ohy
                    ) / dhy - ihy_b  # weight for front psfs, left 1-ihz_t
                    dhz_d = (
                        iz - ohz
                    ) / dhz - ihz_t  # weight for down psfs,  top 1-dhz_d

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

                    h = (
                        dhx_l * dhy_b * dhz_t * h_lbt
                        + dhx_l * dhy_b * dhz_d * h_lbd
                        + dhx_l * dhy_f * dhz_t * h_lft
                        + dhx_l * dhy_f * dhz_d * h_lfd
                        + dhx_r * dhy_b * dhz_t * h_rbt
                        + dhx_r * dhy_b * dhz_d * h_rbd
                        + dhx_r * dhy_f * dhz_t * h_rft
                        + dhx_r * dhy_f * dhz_d * h_rfd
                    )

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

                    if not rmatvec:
                        y[
                            xextremes[0] : xextremes[1],
                            yextremes[0] : yextremes[1],
                            zextremes[0] : zextremes[1],
                        ] += (
                            x[ix, iy, iz]
                            * h[
                                hxextremes[0] : hxextremes[1],
                                hyextremes[0] : hyextremes[1],
                                hzextremes[0] : hzextremes[1],
                            ]
                        )
                    else:
                        y[ix, iy, iz] = np.sum(
                            h[
                                hxextremes[0] : hxextremes[1],
                                hyextremes[0] : hyextremes[1],
                                hzextremes[0] : hzextremes[1],
                            ]
                            * x[
                                xextremes[0] : xextremes[1],
                                yextremes[0] : yextremes[1],
                                zextremes[0] : zextremes[1],
                            ]
                        )
        return y

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(self.dims, dtype=self.dtype)
        y = self._mvrmv(
            x,
            y,
            self.hs,
            self.hshape,
            self.dims,
            self.ohx,
            self.ohy,
            self.ohz,
            self.dhx,
            self.dhy,
            self.dhz,
            self.nhx,
            self.nhy,
            self.nhz,
            rmatvec=False,
            **self.kwargs_cuda
        )
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(self.dims, dtype=self.dtype)
        y = self._mvrmv(
            x,
            y,
            self.hs,
            self.hshape,
            self.dims,
            self.ohx,
            self.ohy,
            self.ohz,
            self.dhx,
            self.dhy,
            self.dhz,
            self.nhx,
            self.nhy,
            self.nhz,
            rmatvec=True,
            **self.kwargs_cuda
        )
        return y
