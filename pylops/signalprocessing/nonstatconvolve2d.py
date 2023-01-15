__all__ = ["NonStationaryConvolve2D"]

import os
from typing import Tuple, Union

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import get_array_module
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

jit_message = deps.numba_import("the nonstatconvolve2d module")

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


class NonStationaryConvolve2D(LinearOperator):
    r"""2D non-stationary convolution operator.

    Apply non-stationary two-dimensional convolution. A varying compact filter
    is provided on a coarser grid and on-the-fly interpolation is applied
    in forward and adjoint modes.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    hs : :obj:`numpy.ndarray`
        Bank of 2d compact filters of size
        :math:`n_{\text{filts},x} \times n_{\text{filts},z} \times n_h \times n_{h,x} \times n_{h,z}`.
        Filters must have odd number of samples and are assumed to be
        centered in the middle of the filter support.
    ihx : :obj:`tuple`
        Indices of the x locations of the filters ``hs`` in the model (and data). Note
        that the filters must be regularly sampled, i.e. :math:`dh_x=\text{diff}(ihx)=\text{const.}`
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
        If ``ihx`` or ``ihz`` is not regularly sampled
    NotImplementedError
        If ``engine`` is neither ``numpy``, ``fftw``, nor ``scipy``.

    Notes
    -----
    The NonStationaryConvolve2D operator applies non-stationary
    two-dimensional convolution between the input signal :math:`d(x, z)`
    and a bank of compact filter kernels :math:`h(x, z; x_i, z_i)`.
    Assuming an input signal composed of :math:`N \times M` samples
    (with :math:`N=4` and :math:`M=3`, and filters at locations :math:`x_1, x_3`
    and :math:`z_1, z_3`, the forward operator can be represented as follows:

    .. math::
        \mathbf{y} =
        \begin{bmatrix}
           \hat{h}_{(0,0),(0,0)} & \cdots & h_{(1,1),(0,0)} & \cdots & \hat{h}_{(2,2),(0,0)} & \cdots  \\
           \hat{h}_{(0,0),(0,1)} & \cdots & h_{(1,1),(0,1)} & \cdots & \hat{h}_{(2,2),(0,0)} & \cdots  \\
           \vdots                & \ddots &                 & \ddots & \vdots                & \vdots  \\
           \hat{h}_{(0,0),(4,3)} & \cdots & h_{(1,1),(4,3)} & \cdots & \hat{h}_{(2,2),(0,0)} & \cdots  \\
        \end{bmatrix}
        \begin{bmatrix}
           x_{0,0} \\ \vdots \\ x_{0,N} \\ x_{1,0} \\ \vdots \\
           x_{1,N} \\ x_{M,0} \\ \vdots \\ x_{M,N}
        \end{bmatrix}

    where :math:`\mathbf{h}_{(1,1)} = [h_{(1,1),(0,0)}, h_{(1,1),(0,1)}, \ldots, h_{(1,1),(4,3)}]`
    (and :math:`\mathbf{h}_{(1,1)}`, :math:`\mathbf{h}_{(1,3)}`, :math:`\mathbf{h}_{(3,1)}`,
    :math:`\mathbf{h}_{(3,3)}`) are the provided filter, :math:`\hat{\mathbf{h}}_{(0,0)} =
    \mathbf{h}_{(1,1)}` and similar are the filters outside the range of the provided filters
    (which are extrapolated to be the same as the nearest provided filter) and
    :math:`\hat{\mathbf{h}}_{(2,2)} = \text{bilinear}(\mathbf{h}_{(1,1)}, \mathbf{h}_{(3,1)},
    \mathbf{h}_{(1,3)},\mathbf{h}_{(3,3)})` is the filter within the range of the provided filters
    (which is bilinearly interpolated from the four nearest provided filter on either side
    of its location).

    For more details on the numerical implementation of the forward and adjoint,
    see :class:`pylops.signalprocessing.NonStationaryConvolve1D`.

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        hs: NDArray,
        ihx: InputDimsLike,
        ihz: InputDimsLike,
        engine: str = "numpy",
        num_threads_per_blocks: Tuple[int, int] = (32, 32),
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        if engine not in ["numpy", "numba", "cuda"]:
            raise NotImplementedError("engine must be numpy or numba or cuda")
        if hs.shape[2] % 2 == 0 or hs.shape[3] % 2 == 0:
            raise ValueError("filters hs must have odd length")
        if len(np.unique(np.diff(ihx))) > 1 or len(np.unique(np.diff(ihz))) > 1:
            raise ValueError(
                "the indices of filters 'ih' are must be regularly sampled"
            )

        self.hs = hs
        self.hshape = hs.shape[2:]
        self.ohx, self.dhx, self.nhx = ihx[0], ihx[1] - ihx[0], len(ihx)
        self.ohz, self.dhz, self.nhz = ihz[0], ihz[1] - ihz[0], len(ihz)
        self.ehx, self.ehz = ihx[-1], ihz[-1]
        self.dims = dims
        self.engine = engine
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

        # create additional input parameters for engine=cuda
        self.kwargs_cuda = {}
        if engine == "cuda":
            self.kwargs_cuda["num_threads_per_blocks"] = num_threads_per_blocks
            num_threads_per_blocks_x, num_threads_per_blocks_z = num_threads_per_blocks
            num_blocks_x = (
                self.dims[0] + num_threads_per_blocks_x - 1
            ) // num_threads_per_blocks_x
            num_blocks_z = (
                self.dims[1] + num_threads_per_blocks_z - 1
            ) // num_threads_per_blocks_z
            self.kwargs_cuda["num_blocks"] = (num_blocks_x, num_blocks_z)
        self._register_multiplications(engine)

    def _register_multiplications(self, engine: str) -> None:
        if engine == "numba":
            numba_opts = dict(nopython=True, fastmath=True, nogil=True, parallel=True)
            self._mvrmv = staticmethod(jit(**numba_opts)(self._matvec_rmatvec))
        elif engine == "cuda":
            self._mvrmv = staticmethod(_matvec_rmatvec_cuda_call)
        else:
            self._mvrmv = self._matvec_rmatvec

    @staticmethod
    def _matvec_rmatvec(
        x, y, hs, hshape, xdims, ohx, ohz, dhx, dhz, nhx, nhz, rmatvec=False
    ):
        for ix in prange(xdims[0]):
            for iz in range(xdims[1]):
                # find closest filters and interpolate h
                ihx_l = int(np.floor((ix - ohx) / dhx))
                ihz_t = int(np.floor((iz - ohz) / dhz))
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

                h = (
                    dhz_t * dhx_l * h_tl
                    + dhz_b * dhx_l * h_bl
                    + dhz_t * dhx_r * h_tr
                    + dhz_b * dhx_r * h_br
                )

                # find extremes of model where to apply h (in case h is going out of model)
                xextremes = (
                    max(0, ix - hshape[0] // 2),
                    min(ix + hshape[0] // 2 + 1, xdims[0]),
                )
                zextremes = (
                    max(0, iz - hshape[1] // 2),
                    min(iz + hshape[1] // 2 + 1, xdims[1]),
                )
                # find extremes of h (in case h is going out of model)
                hxextremes = (
                    max(0, -ix + hshape[0] // 2),
                    min(hshape[0], hshape[0] // 2 + (xdims[0] - ix)),
                )
                hzextremes = (
                    max(0, -iz + hshape[1] // 2),
                    min(hshape[1], hshape[1] // 2 + (xdims[1] - iz)),
                )
                if not rmatvec:
                    y[xextremes[0] : xextremes[1], zextremes[0] : zextremes[1]] += (
                        x[ix, iz]
                        * h[
                            hxextremes[0] : hxextremes[1], hzextremes[0] : hzextremes[1]
                        ]
                    )
                else:
                    y[ix, iz] = np.sum(
                        h[hxextremes[0] : hxextremes[1], hzextremes[0] : hzextremes[1]]
                        * x[xextremes[0] : xextremes[1], zextremes[0] : zextremes[1]]
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
            self.ohz,
            self.dhx,
            self.dhz,
            self.nhx,
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
            self.ohz,
            self.dhx,
            self.dhz,
            self.nhx,
            self.nhz,
            rmatvec=True,
            **self.kwargs_cuda
        )
        return y
