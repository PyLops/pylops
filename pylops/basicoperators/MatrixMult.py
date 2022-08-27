import logging
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import scipy as sp
from scipy.sparse.linalg import inv

from pylops import LinearOperator
from pylops.utils._internal import _value_or_list_like_to_array
from pylops.utils.backend import get_array_module

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


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
        A: npt.ArrayLike,
        otherdims: Optional[Tuple[int]] = None,
        dtype: str = "float64",
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
            otherdims = _value_or_list_like_to_array(otherdims)
            self.otherdims = np.array(otherdims, dtype=int)
            dims, dimsd = np.insert(self.otherdims, 0, self.A.shape[1]), np.insert(
                self.otherdims, 0, self.A.shape[0]
            )
            self.dimsflatten, self.dimsdflatten = np.insert(
                [np.prod(self.otherdims)], 0, self.A.shape[1]
            ), np.insert([np.prod(self.otherdims)], 0, self.A.shape[0])
            self.reshape = True
            explicit = False

        # Check dtype for correctness (upcast to complex when A is complex)
        if np.iscomplexobj(A) and not np.iscomplexobj(np.ones(1, dtype=dtype)):
            dtype = A.dtype
            logging.warning("Matrix A is a complex object, dtype cast to %s" % dtype)
        super().__init__(
            dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, explicit=explicit, name=name
        )

    def _matvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        ncp = get_array_module(x)
        if self.reshape:
            x = ncp.reshape(x, self.dimsflatten)
        y = self.A.dot(x)
        if self.reshape:
            return y.ravel()
        else:
            return y

    def _rmatvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
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

    def inv(self) -> npt.ArrayLike:
        r"""Return the inverse of :math:`\mathbf{A}`.

        Returns
        ----------
        Ainv : :obj:`numpy.ndarray`
            Inverse matrix.

        """
        if sp.sparse.issparse(self.A):
            Ainv = inv(self.A)
        else:
            ncp = get_array_module(self.A)
            Ainv = ncp.linalg.inv(self.A)
        return Ainv
