__all__ = [
    "NonStationaryConvolve1D",
    "NonStationaryFilters1D",
]

from typing import Union

import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import get_array_module
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class NonStationaryConvolve1D(LinearOperator):
    r"""1D non-stationary convolution operator.

    Apply non-stationary one-dimensional convolution. A varying compact filter
    is provided on a coarser grid and on-the-fly interpolation is applied
    in forward and adjoint modes.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    hs : :obj:`numpy.ndarray`
        Bank of 1d compact filters of size :math:`n_\text{filts} \times n_h`.
        Filters must have odd number of samples and are assumed to be
        centered in the middle of the filter support.
    ih : :obj:`tuple`
        Indices of the locations of the filters ``hs`` in the model (and data). Note
        that the filters must be regularly sampled, i.e. :math:`dh=\text{diff}(ih)=\text{const.}`
    axis : :obj:`int`, optional
        Axis along which convolution is applied
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
        If ``ih`` is not regularly sampled

    Notes
    -----
    The NonStationaryConvolve1D operator applies non-stationary
    one-dimensional convolution between the input signal :math:`d(t)`
    and a bank of compact filter kernels :math:`h(t; t_i)`. Assuming
    an input signal composed of :math:`N=5` samples, and filters at locations
    :math:`t_1` and :math:`t_3`, the forward operator can be represented as follows:

    .. math::
        \mathbf{y} =
        \begin{bmatrix}
           \hat{h}_{0,0} & h_{1,0} & \hat{h}_{2,0} & h_{3,0} & \hat{h}_{4,0} \\
           \hat{h}_{0,1} & h_{1,1} & \hat{h}_{2,1} & h_{3,1} & \hat{h}_{4,1} \\
           \vdots        & \vdots  & \vdots        & \vdots  & \vdots        \\
           \hat{h}_{0,4} & h_{1,4} & \hat{h}_{2,4} & h_{3,4} & \hat{h}_{4,4} \\
        \end{bmatrix}
        \begin{bmatrix}
           x_0 \\ x_1 \\ \vdots \\ x_4
        \end{bmatrix}

    where :math:`\mathbf{h}_1 = [h_{1,0}, h_{1,1}, \ldots, h_{1,N}]` and
    :math:`\mathbf{h}_3 = [h_{3,0}, h_{3,1}, \ldots, h_{3,N}]` are the provided filter,
    :math:`\hat{\mathbf{h}}_0 = \mathbf{h}_1` and :math:`\hat{\mathbf{h}}_4 = \mathbf{h}_3` are the
    filters outside the range of the provided filters (which are extrapolated to be the same as
    the nearest provided filter) and :math:`\hat{\mathbf{h}}_2 = 0.5 \mathbf{h}_1 + 0.5 \mathbf{h}_3`
    is the filter within the range of the provided filters (which is linearly interpolated from the two nearest
    provided filter on either side of its location).

    In forward mode, each filter is weighted by the corresponding input and spread over the output.
    In adjoint mode, the corresponding filter is element-wise multiplied with the input, all values
    are summed together and put in the output.

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        hs: NDArray,
        ih: InputDimsLike,
        axis: int = -1,
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        if hs.shape[1] % 2 == 0:
            raise ValueError("filters hs must have odd length")
        if len(np.unique(np.diff(ih))) > 1:
            raise ValueError(
                "the indices of filters 'ih' are must be regularly sampled"
            )
        self.hs = hs
        self.hsize = hs.shape[1]
        self.oh, self.dh, self.nh, self.eh = ih[0], ih[1] - ih[0], len(ih), ih[-1]
        self.axis = axis

        dims = _value_or_sized_to_tuple(dims)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

    @property
    def hsinterp(self):
        ncp = get_array_module(self.hs)
        self._hsinterp = ncp.empty((self.dims[self.axis], self.hsize), dtype=self.dtype)

        for ix in range(self.dims[self.axis]):
            self._hsinterp[ix] = self._interpolate_h(
                self.hs, ix, self.oh, self.dh, self.nh
            )
        return self._hsinterp

    @hsinterp.deleter
    def hsinterp(self):
        del self._hsinterp

    @staticmethod
    def _interpolate_h(hs, ix, oh, dh, nh):
        """find closest filters and linearly interpolate between them and interpolate psf"""
        ih_closest = int(np.floor((ix - oh) / dh))
        if ih_closest < 0:
            h = hs[0]
        elif ih_closest >= nh - 1:
            h = hs[nh - 1]
        else:
            dh_closest = (ix - oh) / dh - ih_closest
            h = (1 - dh_closest) * hs[ih_closest] + dh_closest * hs[ih_closest + 1]
        return h

    @reshaped(swapaxis=True)
    def _matvec(self, x: NDArray) -> NDArray:
        y = np.zeros_like(x)
        for ix in range(self.dims[self.axis]):
            h = self._interpolate_h(self.hs, ix, self.oh, self.dh, self.nh)
            xextremes = (
                max(0, ix - self.hsize // 2),
                min(ix + self.hsize // 2 + 1, self.dims[self.axis]),
            )
            hextremes = (
                max(0, -ix + self.hsize // 2),
                min(self.hsize, self.hsize // 2 + (self.dims[self.axis] - ix)),
            )
            y[..., xextremes[0] : xextremes[1]] += (
                x[..., ix : ix + 1] * h[hextremes[0] : hextremes[1]]
            )
        return y

    @reshaped(swapaxis=True)
    def _rmatvec(self, x: NDArray) -> NDArray:
        y = np.zeros_like(x)
        for ix in range(self.dims[self.axis]):
            h = self._interpolate_h(self.hs, ix, self.oh, self.dh, self.nh)
            xextremes = (
                max(0, ix - self.hsize // 2),
                min(ix + self.hsize // 2 + 1, self.dims[self.axis]),
            )
            hextremes = (
                max(0, -ix + self.hsize // 2),
                min(self.hsize, self.hsize // 2 + (self.dims[self.axis] - ix)),
            )
            y[..., ix] = np.sum(
                h[hextremes[0] : hextremes[1]] * x[..., xextremes[0] : xextremes[1]],
                axis=-1,
            )
        return y

    def todense(self):
        hs = self.hsinterp
        H = np.array(
            [
                np.roll(np.pad(h, (0, self.dims[self.axis])), ix)
                for ix, h in enumerate(hs)
            ]
        )
        H = H[:, int(self.hsize // 2) : -int(self.hsize // 2) - 1]
        return H


class NonStationaryFilters1D(LinearOperator):
    r"""1D non-stationary filter estimation operator.

    Estimate a non-stationary one-dimensional filter by non-stationary convolution.
    In forward mode, a varying compact filter on a coarser grid is on-the-fly linearly interpolated prior
    to being convolved with a fixed input signal. In adjoint mode, the output signal is first weighted by the
    fixed input signal and then spread across multiple filters (i.e., adjoint of linear interpolation).

    Parameters
    ----------
    inp : :obj:`numpy.ndarray`
        Fixed input signal of size :math:`n_x`.
    hsize : :obj:`int`
        Size of the 1d compact filters (filters must have odd number of samples and are assumed to be
        centered in the middle of the filter support).
    ih : :obj:`tuple`
        Indices of the locations of the filters ``hs`` in the model (and data). Note
        that the filters must be regularly sampled, i.e. :math:`dh=\text{diff}(ih)=\text{const.}`
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
        If filters ``hsize`` is a even number
    ValueError
        If ``ih`` is not regularly sampled

    See Also
    --------
    NonStationaryConvolve1D : 1D non-stationary convolution operator.
    NonStationaryFilters2D : 2D non-stationary filter estimation operator.

    Notes
    -----
    The NonStationaryFilters1D operates in a similar fashion to the
    :class:`pylops.signalprocessing.NonStationaryConvolve1D` operator. In practical applications,
    this operator shall be used when interested to estimate a 1-dimensional non-stationary filter
    given an input and output signal.

    In forward mode, this operator uses the same implementation of the
    :class:`pylops.signalprocessing.NonStationaryConvolve1D`, with the main difference that
    the role of the filters and the input signal is swapped. Nevertheless, to understand how
    to implement adjoint, mathematically we arrange the forward operator in a slightly different way.
    Assuming once again an input signal composed of :math:`N=5` samples, and filters at locations
    :math:`t_1` and :math:`t_3`, the forward operator can be represented as follows:

    .. math::
        \mathbf{y} =
        \begin{bmatrix}
           \mathbf{X}_0 & \mathbf{X}_1 & \vdots & \mathbf{X}_4
        \end{bmatrix} \mathbf{L}
        \begin{bmatrix}
            \mathbf{h}_1 \\ \mathbf{h}_3
        \end{bmatrix}

    where :math:`\mathbf{L}` is an operator that linearly interpolates the filters from the available locations to
    the entire input grid -- i.e., :math:`[\hat{\mathbf{h}}_0 \quad \mathbf{h}_1 \quad \hat{\mathbf{h}}_2 \quad
    \mathbf{h}_3 \quad \hat{\mathbf{h}}_4]^T = \mathbf{L} [ \mathbf{h}_1 \quad \mathbf{h}_3]`. Finally,
    :math:`\mathbf{X}_i` is a diagonal matrix containing the value :math:`x_i` along the
    main diagonal. Note that in practice the filter may be shorter than the input and output signals and
    the :math:`x_i` values are placed only at the effective positions of the filter along the diagonal matrices
    :math:`\mathbf{X}_i`.

    In adjoint mode, the output signal is first weighted by the fixed input signal and then spread across
    multiple filters (i.e., adjoint of linear interpolation) as follows

    .. math::
        \mathbf{h} =
        \mathbf{L}^H
        \begin{bmatrix}
           \mathbf{X}_0 \\ \mathbf{X}_1 \\ \vdots \\ \mathbf{X}_4
        \end{bmatrix}
        \mathbf{y}

    """

    def __init__(
        self,
        inp: NDArray,
        hsize: int,
        ih: InputDimsLike,
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        if hsize % 2 == 0:
            raise ValueError("filters hs must have odd length")
        if len(np.unique(np.diff(ih))) > 1:
            raise ValueError(
                "the indices of filters 'ih' are must be regularly sampled"
            )
        self.inp = inp
        self.hsize = hsize
        self.oh, self.dh, self.nh, self.eh = ih[0], ih[1] - ih[0], len(ih), ih[-1]

        super().__init__(
            dtype=np.dtype(dtype), dims=(len(ih), hsize), dimsd=inp.shape, name=name
        )

    # use same interpolation method as inNonStationaryConvolve1D
    _interpolate_h = staticmethod(NonStationaryConvolve1D._interpolate_h)

    @staticmethod
    def _interpolate_hadj(htmp, hs, hextremes, ix, oh, dh, nh):
        """find closest filters and spread weighted psf"""
        ih_closest = int(np.floor((ix - oh) / dh))
        if ih_closest < 0:
            hs[0, hextremes[0] : hextremes[1]] += htmp
        elif ih_closest >= nh - 1:
            hs[nh - 1, hextremes[0] : hextremes[1]] += htmp
        else:
            dh_closest = (ix - oh) / dh - ih_closest
            hs[ih_closest, hextremes[0] : hextremes[1]] += (1 - dh_closest) * htmp
            hs[ih_closest + 1, hextremes[0] : hextremes[1]] += dh_closest * htmp
        return hs

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = np.zeros(self.dimsd, dtype=self.dtype)
        for ix in range(self.dimsd[0]):
            h = self._interpolate_h(x, ix, self.oh, self.dh, self.nh)
            xextremes = (
                max(0, ix - self.hsize // 2),
                min(ix + self.hsize // 2 + 1, self.dimsd[0]),
            )
            hextremes = (
                max(0, -ix + self.hsize // 2),
                min(self.hsize, self.hsize // 2 + (self.dimsd[0] - ix)),
            )
            y[..., xextremes[0] : xextremes[1]] += (
                self.inp[..., ix : ix + 1] * h[hextremes[0] : hextremes[1]]
            )
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        hs = np.zeros(self.dims, dtype=self.dtype)
        for ix in range(self.dimsd[0]):
            xextremes = (
                max(0, ix - self.hsize // 2),
                min(ix + self.hsize // 2 + 1, self.dimsd[0]),
            )
            hextremes = (
                max(0, -ix + self.hsize // 2),
                min(self.hsize, self.hsize // 2 + (self.dimsd[0] - ix)),
            )

            htmp = self.inp[ix] * x[..., xextremes[0] : xextremes[1]]
            hs = self._interpolate_hadj(
                htmp, hs, hextremes, ix, self.oh, self.dh, self.nh
            )
        return hs
