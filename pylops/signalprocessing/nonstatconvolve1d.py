__all__ = ["NonStationaryConvolve1D"]

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
    an input signal composed of :math:`N=4` samples, and filters at locations
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
    ) -> LinearOperator:
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
        _hsinterp = ncp.empty((self.dims[self.axis], self.hsize), dtype=self.dtype)

        for ix in range(self.dims[self.axis]):
            _hsinterp[ix] = self._interpolate_h(self.hs, ix, self.oh, self.dh, self.nh)
        return _hsinterp

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
