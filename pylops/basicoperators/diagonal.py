__all__ = ["Diagonal"]

from typing import List, Optional

import numpy as np
import numpy.typing as npt

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import get_array_module, to_cupy_conditional
from pylops.utils.decorators import reshaped


class Diagonal(LinearOperator):
    r"""Diagonal operator.

    Applies element-wise multiplication of the input vector with the vector
    ``diag`` in forward and with its complex conjugate in adjoint mode.

    This operator can also broadcast; in this case the input vector is
    reshaped into its dimensions ``dims`` and the element-wise multiplication
    with ``diag`` is perfomed along ``axis``. Note that the
    vector ``diag`` will need to have size equal to ``dims[axis]``.

    Parameters
    ----------
    diag : :obj:`numpy.ndarray`
        Vector to be used for element-wise multiplication.
    dims : :obj:`list`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which multiplication is applied.
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

    Notes
    -----
    Element-wise multiplication between the model :math:`\mathbf{x}` and/or
    data :math:`\mathbf{y}` vectors and the array :math:`\mathbf{d}`
    can be expressed as

    .. math::

        y_i = d_i x_i  \quad \forall i=1,2,\ldots,N

    This is equivalent to a matrix-vector multiplication with a matrix
    containing the vector :math:`\mathbf{d}` along its main diagonal.

    For real-valued ``diag``, the Diagonal operator is self-adjoint as the
    adjoint of a diagonal matrix is the diagonal matrix itself. For
    complex-valued ``diag``, the adjoint is equivalent to the element-wise
    multiplication with the complex conjugate elements of ``diag``.

    """

    def __init__(
        self,
        diag: npt.ArrayLike,
        dims: Optional[List[int]] = None,
        axis: int = -1,
        dtype: str = "float64",
        name: str = "D",
    ) -> None:
        self.diag = diag.ravel()
        dims = (len(self.diag),) if dims is None else _value_or_sized_to_tuple(dims)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

        ncp = get_array_module(diag)
        self.complex = True if ncp.iscomplexobj(self.diag) else False
        diagdims = np.ones_like(self.dims)
        diagdims[axis] = self.dims[axis]
        self.diag = self.diag.reshape(diagdims)

    @reshaped
    def _matvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        if type(self.diag) != type(x):
            self.diag = to_cupy_conditional(x, self.diag)
        y = self.diag * x
        return y

    @reshaped
    def _rmatvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        if type(self.diag) != type(x):
            self.diag = to_cupy_conditional(x, self.diag)
        if self.complex:
            diagadj = self.diag.conj()
        else:
            diagadj = self.diag
        y = diagadj * x
        return y

    def matrix(self) -> npt.ArrayLike:
        """Return diagonal matrix as dense :obj:`numpy.ndarray`

        Returns
        -------
        densemat : :obj:`numpy.ndarray`
            Dense matrix.

        """
        ncp = get_array_module(self.diag)
        densemat = ncp.diag(self.diag.squeeze())
        return densemat

    def todense(self) -> npt.ArrayLike:
        """Fast implementation of todense based on known structure of the
        operator

        Returns
        -------
        densemat : :obj:`numpy.ndarray`
            Dense matrix.
        """
        return self.matrix()
