__all__ = ["Restriction"]

import logging
from typing import Sequence, Union

import numpy as np
import numpy.ma as np_ma

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import (
    get_array_module,
    get_normalize_axis_index,
    inplace_set,
    to_numpy,
)
from pylops.utils.typing import DTypeLike, InputDimsLike, IntNDArray, NDArray

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def _compute_iavamask(dims, axis, iava, ncp):
    """Compute restriction mask when using cupy arrays"""
    otherdims = np.array(dims)
    otherdims = np.delete(otherdims, axis)
    iavamask = np.zeros(int(dims[axis]), dtype=int)
    iavamask[iava] = 1
    iavamask = np.moveaxis(
        np.broadcast_to(iavamask, list(otherdims) + [dims[axis]]), -1, axis
    )
    iavamask = np.where(iavamask.ravel() == 1)[0]
    return ncp.asarray(iavamask)


class Restriction(LinearOperator):
    r"""Restriction (or sampling) operator.

    Extract subset of values from input vector at locations ``iava``
    in forward mode and place those values at locations ``iava``
    in an otherwise zero vector in adjoint mode.

    Parameters
    ----------
    dims : :obj:`list` or :obj:`int`
        Number of samples for each dimension
    iava : :obj:`list` or :obj:`numpy.ndarray`
        Integer indices of available samples for data selection.
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which restriction is applied to model.
    inplace : :obj:`bool`, optional
        Work inplace (``True``) or make a new copy (``False``). By default,
        data is a reference to the model (in forward) and model is a reference
        to the data (in adjoint).
    forceflat : :obj:`bool`, optional
        .. versionadded:: 2.2.0

        Force an array to be flattened after rmatvec. Note that this is only
        required when `len(dims)=2`, otherwise pylops will detect whether to
        return a 1d or nd array.
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
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    See Also
    --------
    pylops.signalprocessing.Interp : Interpolation operator

    Notes
    -----
    Extraction (or *sampling*) of a subset of :math:`N` values at locations
    ``iava`` from an input (or model) vector :math:`\mathbf{x}` of size
    :math:`M` can be expressed as:

    .. math::

        y_i = x_{l_i}  \quad \forall i=0,1,\ldots,N-1

    where :math:`\mathbf{l}=[l_0, l_1,\ldots, l_{N-1}]` is a vector containing the indices
    of the original array at which samples are taken.

    Conversely, in adjoint mode the available values in the data vector
    :math:`\mathbf{y}` are placed at locations
    :math:`\mathbf{l}=[l_0, l_1,\ldots, l_{M-1}]` in the model vector:

    .. math::

        x_{l_i} = y_i  \quad \forall i=0,1,\ldots,N-1

    and :math:`x_{j}=0` for :math:`j \neq l_i` (i.e., at all other locations in input
    vector).

    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        iava: Union[IntNDArray, Sequence[int]],
        axis: int = -1,
        inplace: bool = True,
        forceflat: bool = None,
        dtype: DTypeLike = "float64",
        name: str = "R",
    ) -> None:
        ncp = get_array_module(iava)
        dims = _value_or_sized_to_tuple(dims)
        axis = get_normalize_axis_index()(axis, len(dims))
        dimsd = list(dims)  # data dimensions
        dimsd[axis] = len(iava)

        # check if forceflat is needed and set it back to None otherwise
        if len(dims) > 2:
            if forceflat is not None:
                logging.warning(
                    f"setting forceflat=None since len(dims)={len(dims)}>2. "
                    f"PyLops will automatically detect whether to return "
                    f"a 1d or nd array based on the shape of the input"
                    f"array."
                )
                forceflat = None

        super().__init__(
            dtype=np.dtype(dtype),
            dims=dims,
            dimsd=dimsd,
            forceflat=forceflat,
            name=name,
        )

        iavareshape = np.ones(len(self.dims), dtype=int)
        iavareshape[axis] = len(iava)

        # currently cupy does not support put_along_axis, so we need to
        # explicitly create a list of indices in the n-dimensional
        # model space which will be used in _rmatvec to place the input
        if ncp != np:
            self.iavamask = _compute_iavamask(self.dims, axis, to_numpy(iava), ncp)
        self.inplace = inplace
        self.axis = axis
        self.iavareshape = iavareshape
        self.iava = ncp.asarray(iava)

    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if not self.inplace:
            x = x.copy()
        x = ncp.reshape(x, self.dims)
        y = ncp.take(x, self.iava, axis=self.axis)
        y = y.ravel()
        return y

    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if not self.inplace:
            x = x.copy()
        x = ncp.reshape(x, self.dimsd)
        if ncp == np:
            y = ncp.zeros(self.dims, dtype=self.dtype)
            ncp.put_along_axis(
                y, ncp.reshape(self.iava, self.iavareshape), x, axis=self.axis
            )
        else:
            if not hasattr(self, "iavamask"):
                self.iavamask = _compute_iavamask(self.dims, self.axis, self.iava, ncp)
            y = ncp.zeros(int(self.shape[-1]), dtype=self.dtype)
            y = inplace_set(x.ravel(), y, self.iavamask)
        y = y.ravel()
        return y

    def mask(self, x: NDArray) -> NDArray:
        """Apply mask to input signal returning a signal of same size with
        values at ``iava`` locations and ``0`` at other locations

        Parameters
        ----------
        x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
            Input array (can be either flattened or not)

        Returns
        -------
        y : :obj:`numpy.ma.core.MaskedArray`
            Masked array.

        """
        ncp = get_array_module(x)
        if ncp != np:
            iava = ncp.asnumpy(self.iava)
        else:
            iava = self.iava.copy()

        y = np_ma.array(np.zeros(self.dims), mask=np.ones(self.dims), dtype=self.dtype)
        x = np.reshape(x, self.dims)
        x = np.swapaxes(x, self.axis, -1)
        y = np.swapaxes(y, self.axis, -1)
        y.mask[..., iava] = False
        if ncp == np:
            y[..., iava] = x[..., self.iava]
        else:
            y[..., iava] = ncp.asnumpy(x)[..., iava]
        y = np.swapaxes(y, -1, self.axis)
        return y
