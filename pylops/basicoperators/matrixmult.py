__all__ = ["MatrixMult"]

import logging
import warnings
from typing import Optional, Union

import numpy as np
import scipy as sp
from scipy.sparse.linalg import inv

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_array
from pylops.utils.backend import get_array_module
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray

logger = logging.getLogger(__name__)


class MatrixMult(LinearOperator):
    r"""Matrix multiplication.

    Simple wrapper to :py:func:`numpy.dot` and :py:func:`numpy.vdot` for
    an input matrix :math:`\mathbf{A}`.

    Parameters
    ----------
    A : :obj:`numpy.ndarray` or :obj:`scipy.sparse` matrix
        Matrix.
    otherdims : :obj:`tuple`, optional
        Number of samples for each other dimension of model
        (model/data will be reshaped and ``A`` applied multiple times
        to each column of the model/data).
    forceflat : :obj:`bool`, optional
         .. versionadded:: 2.2.0

         Force an array to be flattened after matvec and rmatvec. Note that this is only
         required when `otherdims=None`, otherwise pylops will detect whether to
         return a 1d or nd array.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before linearization.

        For example, ``y_reshaped = (Op * x.ravel()).reshape(Op.dimsd)``.
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)
    complex : :obj:`bool`
        Matrix has complex numbers (``True``) or not (``False``)

    """

    def __init__(
        self,
        A: NDArray,
        otherdims: Optional[Union[int, InputDimsLike]] = None,
        forceflat: bool = None,
        dtype: DTypeLike = "float64",
        name: str = "M",
    ) -> None:
        ncp = get_array_module(A)
        self.A = A
        if isinstance(A, ncp.ndarray):
            self.complex = np.iscomplexobj(A)
        else:
            self.complex = np.iscomplexobj(A.data)
        if otherdims is None:
            dims, dimsd = (A.shape[1],), (A.shape[0],)
            self.reshape = False
            explicit = True
        else:
            otherdims = _value_or_sized_to_array(otherdims)
            self.otherdims = np.array(otherdims, dtype=int)
            dims, dimsd = (
                np.insert(self.otherdims, 0, self.A.shape[1]),
                np.insert(self.otherdims, 0, self.A.shape[0]),
            )
            self.dimsflatten, self.dimsdflatten = (
                np.insert([np.prod(self.otherdims)], 0, self.A.shape[1]),
                np.insert([np.prod(self.otherdims)], 0, self.A.shape[0]),
            )
            self.reshape = True
            explicit = False

        # Check if forceflat is needed and set it back to None otherwise
        if otherdims is not None and forceflat is not None:
            logger.warning(
                "Setting forceflat=None since otherdims!=None. "
                "PyLops will automatically detect whether to return "
                "a 1d or nd array based on the shape of the input "
                "array."
            )
            forceflat = None
        # Check dtype for correctness (upcast to complex when A is complex)
        if np.iscomplexobj(A) and not np.iscomplexobj(np.ones(1, dtype=dtype)):
            dtype = A.dtype
            warnings.warn("Matrix A is a complex object, dtype cast to %s" % dtype)
        super().__init__(
            dtype=np.dtype(dtype),
            dims=dims,
            dimsd=dimsd,
            explicit=explicit,
            forceflat=forceflat,
            name=name,
        )

    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if self.reshape:
            x = ncp.reshape(x, self.dimsflatten)
        y = self.A.dot(x)
        if self.reshape:
            return y.ravel()
        else:
            return y

    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if self.reshape:
            x = ncp.reshape(x, self.dimsdflatten)
        if self.complex:
            y = (self.A.T.dot(x.conj())).conj()
        else:
            y = self.A.T.dot(x)

        if self.reshape:
            return y.ravel()
        else:
            return y

    def inv(self) -> NDArray:
        r"""Return the inverse of :math:`\mathbf{A}`.

        Returns
        -------
        Ainv : :obj:`numpy.ndarray`
            Inverse matrix.

        """
        if sp.sparse.issparse(self.A):
            Ainv = inv(self.A)
        else:
            ncp = get_array_module(self.A)
            Ainv = ncp.linalg.inv(self.A)
        return Ainv
