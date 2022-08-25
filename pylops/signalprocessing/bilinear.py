__all__ = ["Bilinear"]

import logging

import numpy as np
import numpy.typing as npt

from pylops import LinearOperator
from pylops.utils.backend import get_add_at, get_array_module, to_numpy
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, IntNDArray, NDArray

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def _checkunique(iava: npt.ArrayLike) -> None:
    """Check that vector as only unique values"""
    _, count = np.unique(iava, axis=1, return_counts=True)
    if np.any(count > 1):
        raise ValueError("Repeated values in iava array")


class Bilinear(LinearOperator):
    r"""Bilinear interpolation operator.

    Apply bilinear interpolation onto fractionary positions ``iava``
    along the first two axes of a n-dimensional array.

    .. note:: The vector ``iava`` should contain unique pais. If the same
       pair is repeated twice an error will be raised.

    Parameters
    ----------
    iava : :obj:`list` or :obj:`numpy.ndarray`
         Array of size :math:`[2 \times n_\text{ava}]` containing
         pairs of floating indices of locations of available samples
         for interpolation.
    dims : :obj:`list`
        Number of samples for each dimension
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

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
        If the vector ``iava`` contains repeated values.

    Notes
    -----
    Bilinear interpolation of a subset of :math:`N` values at locations
    ``iava`` from an input n-dimensional vector :math:`\mathbf{x}` of size
    :math:`[m_1 \times m_2 \times ... \times m_{ndim}]` can be expressed as:

    .. math::

        y_{\mathbf{i}} = (1-w^0_{i}) (1-w^1_{i}) x_{l^{l,0}_i, l^{l,1}_i} +
            w^0_{i} (1-w^1_{i}) x_{l^{r,0}_i, l^{l,1}_i} +
            (1-w^0_{i}) w^1_{i} x_{l^{l,0}_i, l^{r,1}_i} +
            w^0_{i} w^1_{i} x_{l^{r,0}_i, l^{r,1}_i}
        \quad \forall i=1,2,\ldots,M

    where :math:`\mathbf{l^{l,0}}=[\lfloor l_1^0 \rfloor,
    \lfloor l_2^0 \rfloor, ..., \lfloor l_N^0 \rfloor]`,
    :math:`\mathbf{l^{l,1}}=[\lfloor l_1^1 \rfloor,
    \lfloor l_2^1 \rfloor, ..., \lfloor l_N^1 \rfloor]`,
    :math:`\mathbf{l^{r,0}}=[\lfloor l_1^0 \rfloor + 1,
    \lfloor l_2^0 \rfloor + 1, ..., \lfloor l_N^0 \rfloor + 1]`,
    :math:`\mathbf{l^{r,1}}=[\lfloor l_1^1 \rfloor + 1,
    \lfloor l_2^1 \rfloor + 1, ..., \lfloor l_N^1 \rfloor + 1]`,
    are vectors containing the indices of the original array at which samples
    are taken, and :math:`\mathbf{w^j}=[l_1^i - \lfloor l_1^i \rfloor,
    l_2^i - \lfloor l_2^i \rfloor, ..., l_N^i - \lfloor l_N^i \rfloor]`
    (:math:`\forall j=0,1`) are the bilinear interpolation weights.

    """

    def __init__(
        self,
        iava: IntNDArray,
        dims: InputDimsLike,
        dtype: DTypeLike = "float64",
        name: str = "B",
    ) -> None:
        # define dimension of data
        ndims = len(dims)
        dimsd = [len(iava[1])] + list(dims[2:])
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)

        ncp = get_array_module(iava)
        # check non-unique pairs (works only with numpy arrays)
        _checkunique(to_numpy(iava))

        # find indices and weights
        self.iava_t = ncp.floor(iava[0]).astype(int)
        self.iava_b = self.iava_t + 1
        self.weights_tb = iava[0] - self.iava_t
        self.iava_l = ncp.floor(iava[1]).astype(int)
        self.iava_r = self.iava_l + 1
        self.weights_lr = iava[1] - self.iava_l

        # expand dims to weights for nd-arrays
        if ndims > 2:
            for _ in range(ndims - 2):
                self.weights_tb = ncp.expand_dims(self.weights_tb, axis=-1)
                self.weights_lr = ncp.expand_dims(self.weights_lr, axis=-1)

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        return (
            x[self.iava_t, self.iava_l] * (1 - self.weights_tb) * (1 - self.weights_lr)
            + x[self.iava_t, self.iava_r] * (1 - self.weights_tb) * self.weights_lr
            + x[self.iava_b, self.iava_l] * self.weights_tb * (1 - self.weights_lr)
            + x[self.iava_b, self.iava_r] * self.weights_tb * self.weights_lr
        )

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        ncp_add_at = get_add_at(x)
        y = ncp.zeros(self.dims, dtype=self.dtype)
        ncp_add_at(
            y,
            tuple([self.iava_t, self.iava_l]),
            x * (1 - self.weights_tb) * (1 - self.weights_lr),
        )
        ncp_add_at(
            y,
            tuple([self.iava_t, self.iava_r]),
            x * (1 - self.weights_tb) * self.weights_lr,
        )
        ncp_add_at(
            y,
            tuple([self.iava_b, self.iava_l]),
            x * self.weights_tb * (1 - self.weights_lr),
        )
        ncp_add_at(
            y, tuple([self.iava_b, self.iava_r]), x * self.weights_tb * self.weights_lr
        )
        return y
