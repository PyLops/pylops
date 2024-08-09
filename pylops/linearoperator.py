from __future__ import annotations, division

__all__ = [
    "LinearOperator",
    "aslinearoperator",
]

import logging
from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
from numpy.linalg import solve as np_solve
from scipy.linalg import eigvals, lstsq
from scipy.linalg import solve as sp_solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator as spLinearOperator
from scipy.sparse.linalg import eigs as sp_eigs
from scipy.sparse.linalg import eigsh as sp_eigsh
from scipy.sparse.linalg import lobpcg as sp_lobpcg
from scipy.sparse.linalg import lsqr, spsolve

# need to check scipy version since the interface submodule changed into
# _interface from scipy>=1.8.0
sp_version = sp.__version__.split(".")
if int(sp_version[0]) <= 1 and int(sp_version[1]) < 8:
    from scipy.sparse.sputils import isintlike, isshape
else:
    from scipy.sparse._sputils import isintlike, isshape

from typing import Callable, List, Optional, Sequence, Union

from pylops import get_ndarray_multiplication
from pylops.optimization.basic import cgls
from pylops.utils.backend import get_array_module, get_module, get_sparse_eye
from pylops.utils.decorators import count
from pylops.utils.estimators import trace_hutchinson, trace_hutchpp, trace_nahutchpp
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, ShapeLike

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class _LinearOperator(ABC):
    """Meta-class for Linear operator"""

    @abstractmethod
    def _matvec(self, x: NDArray) -> NDArray:
        """Matrix-vector multiplication handler."""
        pass

    @abstractmethod
    def _rmatvec(self, x: NDArray) -> NDArray:
        """Matrix-vector adjoint multiplication handler."""
        pass


class LinearOperator(_LinearOperator):
    """Common interface for performing matrix-vector products.

    This class acts as an abstract interface between matrix-like
    objects and iterative solvers, providing methods to perform
    matrix-vector and adjoint matrix-vector products as as
    well as convenience methods such as ``eigs``, ``cond``, and
    ``conj``.

    .. note:: End users of PyLops should not use this class directly but simply
      use operators that are already implemented. This class is meant for
      developers and it has to be used as the parent class of any new operator
      developed within PyLops. Find more details regarding implementation of
      new operators at :ref:`addingoperator`.

    Parameters
    ----------
    Op : :obj:`scipy.sparse.linalg.LinearOperator` or :obj:`pylops.linearoperator.LinearOperator`
        Operator. If other arguments are provided, they will overwrite those obtained from ``Op``.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    shape : :obj:`tuple(int, int)`, optional
        Shape of operator. If not provided, obtained from ``dims`` and ``dimsd``.
    dims : :obj:`tuple(int, ..., int)`, optional
        .. versionadded:: 2.0.0

        Dimensions of model. If not provided, ``(self.shape[1],)`` is used.
    dimsd : :obj:`tuple(int, ..., int)`, optional
        .. versionadded:: 2.0.0

        Dimensions of data. If not provided, ``(self.shape[0],)`` is used.
    clinear : :obj:`bool`, optional
        .. versionadded:: 1.17.0

        Operator is complex-linear. Defaults to ``True``.
    explicit : :obj:`bool`, optional
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``). Defaults to ``False``.
    forceflat : :obj:`bool`, optional
        .. versionadded:: 2.2.0

        Force an array to be flattened after matvec/rmatvec if the input is ambiguous
        (i.e., is a 1D array both when operating with ND arrays and with 1D arrays).
        Defaults to ``None`` for operators that have no ambiguity or to ``True``
        for those with ambiguity.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    """

    def __init__(
        self,
        Op: Optional[Union[spLinearOperator, LinearOperator]] = None,
        dtype: Optional[DTypeLike] = None,
        shape: Optional[ShapeLike] = None,
        dims: Optional[ShapeLike] = None,
        dimsd: Optional[ShapeLike] = None,
        clinear: Optional[bool] = None,
        explicit: Optional[bool] = None,
        forceflat: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> None:
        if Op is not None:
            self.Op = Op
            # All Operators must have shape and dtype
            dtype = self.Op.dtype if dtype is None else dtype
            shape = self.Op.shape if shape is None else shape
            # Optionally, some operators have other attributes
            dims = getattr(Op, "dims", (Op.shape[1],)) if dims is None else dims
            dimsd = getattr(Op, "dimsd", (Op.shape[0],)) if dimsd is None else dimsd
            clinear = getattr(Op, "clinear", True) if clinear is None else clinear
            explicit = (
                getattr(self.Op, "explicit", False) if explicit is None else explicit
            )
            if explicit and hasattr(Op, "A"):
                self.A = Op.A
            forceflat = (
                getattr(self.Op, "forceflat", None) if forceflat is None else forceflat
            )
            name = getattr(Op, "name", None) if name is None else name

        if dtype is not None:
            self.dtype = dtype
        if shape is not None:
            self.shape = shape
        if dims is not None:
            self.dims = dims
        if dimsd is not None:
            self.dimsd = dimsd
        if clinear is not None:
            self.clinear = clinear
        if explicit is not None:
            self.explicit = explicit
        if forceflat is not None:
            self.forceflat = forceflat
        self.name = name

        # counters
        self.matvec_count = 0
        self.rmatvec_count = 0
        self.matmat_count = 0
        self.rmatmat_count = 0

    @property
    def shape(self):
        _shape = getattr(self, "_shape", None)
        if _shape is None:  # Cannot find shape, falling back on dims and dimsd
            dims = getattr(self, "_dims", None)
            dimsd = getattr(self, "_dimsd", None)
            if dims is None or dimsd is None:  # Cannot find both dims and dimsd, error
                raise AttributeError(
                    (
                        f"'{self.__class__.__name__}' object has no attribute 'shape' "
                        "nor both fallback attributes ('dims', 'dimsd')"
                    )
                )
            _shape = (int(np.prod(dimsd)), int(np.prod(dims)))
            self._shape = _shape  # Update to not redo everything above on next call
        return _shape

    @shape.setter
    def shape(self, new_shape: ShapeLike) -> None:
        new_shape = tuple(new_shape)
        if not isshape(new_shape):
            raise ValueError(f"invalid shape %{new_shape:r} (must be 2-d)")
        dims = getattr(self, "_dims", None)
        dimsd = getattr(self, "_dimsd", None)
        if dims is not None and dimsd is not None:  # Found dims and dimsd
            if np.prod(dimsd) != new_shape[0] and np.prod(dims) != new_shape[1]:
                raise ValueError("New shape incompatible with dims and dimsd")
            elif np.prod(dimsd) != new_shape[0]:
                raise ValueError("New shape incompatible with dimsd")
            elif np.prod(dims) != new_shape[1]:
                raise ValueError("New shape incompatible with dims")
        self._shape = new_shape

    @shape.deleter
    def shape(self):
        del self._shape

    @property
    def dims(self):
        _dims = getattr(self, "_dims", None)
        if _dims is None:
            shape = getattr(self, "_shape", None)
            if shape is None:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attributes 'dims' or 'shape'"
                )
            _dims = (shape[1],)
        return _dims

    @dims.setter
    def dims(self, new_dims: ShapeLike) -> None:
        new_dims = tuple(new_dims)
        shape = getattr(self, "_shape", None)
        if shape is None:  # shape not set yet
            self._dims = new_dims
        else:
            if np.prod(new_dims) == self.shape[1]:
                self._dims = new_dims
            else:
                raise ValueError("dims incompatible with shape[1]")

    @dims.deleter
    def dims(self):
        del self._dims

    @property
    def dimsd(self):
        _dimsd = getattr(self, "_dimsd", None)
        if _dimsd is None:
            shape = getattr(self, "_shape", None)
            if shape is None:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attributes 'dimsd' or 'shape'"
                )
            _dimsd = (shape[0],)
        return _dimsd

    @dimsd.setter
    def dimsd(self, new_dimsd: ShapeLike) -> None:
        new_dimsd = tuple(new_dimsd)
        shape = getattr(self, "_shape", None)
        if shape is None:  # shape not set yet
            self._dimsd = new_dimsd
        else:
            if np.prod(new_dimsd) == self.shape[0]:
                self._dimsd = new_dimsd
            else:
                raise ValueError("dimsd incompatible with shape[0]")

    @dimsd.deleter
    def dimsd(self):
        del self._dimsd

    @property
    def clinear(self):
        return getattr(self, "_clinear", True)

    @clinear.setter
    def clinear(self, new_clinear: bool) -> None:
        self._clinear = bool(new_clinear)

    @clinear.deleter
    def clinear(self):
        del self._clinear

    @property
    def explicit(self):
        return getattr(self, "_explicit", False)

    @explicit.setter
    def explicit(self, new_explicit: bool) -> None:
        self._explicit = bool(new_explicit)

    @explicit.deleter
    def explicit(self):
        del self._explicit

    @property
    def forceflat(self):
        return getattr(self, "_forceflat", None)

    @forceflat.setter
    def forceflat(self, new_forceflat: bool) -> None:
        # note that this can also be None so we check before forcing bool
        self._forceflat = (
            new_forceflat if new_forceflat is None else bool(new_forceflat)
        )

    @forceflat.deleter
    def forceflat(self):
        del self._forceflat

    @property
    def name(self):
        return getattr(self, "_name", None)

    @name.setter
    def name(self, new_name: str) -> None:
        self._name = new_name

    @name.deleter
    def name(self):
        del self._name

    def __mul__(self, x: Union[float, LinearOperator]) -> LinearOperator:
        return self.dot(x)

    def __matmul__(self, other):
        if np.isscalar(other):
            raise ValueError("Scalar not allowed, use * instead")
        return self.__mul__(other)

    def __rmul__(self, x: float) -> LinearOperator:
        if np.isscalar(x):
            Op = _ScaledLinearOperator(self, x)
            self._copy_attributes(
                Op,
                exclude=[
                    "explicit",
                    "name",
                ],
            )
            Op.explicit = False
            return Op
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        if np.isscalar(other):
            raise ValueError("Scalar not allowed, use * instead")
        return self.__rmul__(other)

    def __pow__(self, p: int) -> LinearOperator:
        if np.isscalar(p):
            Op = _PowerLinearOperator(self, p)
            self._copy_attributes(
                Op,
                exclude=[
                    "explicit",
                    "name",
                ],
            )
            Op.explicit = False
            return Op
        else:
            return NotImplemented

    def __add__(self, x: LinearOperator) -> LinearOperator:
        if isinstance(x, (LinearOperator, spLinearOperator)):
            # cast x to pylops linear operator if not already (this is done
            # to allow mixing pylops and scipy operators)
            Opx = aslinearoperator(x)
            Op = _SumLinearOperator(self, Opx)
            self._copy_attributes(
                Op,
                exclude=[
                    "explicit",
                    "forceflat",
                    "name",
                ],
            )
            Op.clinear = Op.clinear and Opx.clinear
            Op.explicit = False
            if self.forceflat is None and Opx.forceflat is None:
                Op.forceflat = None
            elif self.forceflat is not None and Opx.forceflat is not None:
                # Define forceflat only if differing, otherwise raise error
                if self.forceflat != Opx.forceflat:
                    raise ValueError(
                        f"Operators have conflicting forceflat {Op.forceflat} != {Opx.forceflat}"
                    )
                Op.forceflat = self.forceflat
            else:  # Only one of them is None
                Op.forceflat = (
                    self.forceflat if self.forceflat is not None else Opx.forceflat
                )

            # Replace if shape-like
            if len(self.dims) == 1:
                Op.dims = Opx.dims
            if len(self.dimsd) == 1:
                Op.dimsd = Opx.dimsd

            return Op
        else:
            return NotImplemented

    def __neg__(self) -> LinearOperator:
        Op = _ScaledLinearOperator(self, -1)
        self._copy_attributes(
            Op,
            exclude=[
                "explicit",
                "name",
            ],
        )
        Op.explicit = False
        return Op

    def __sub__(self, x):
        return self.__add__(-x)

    def __repr__(self):
        M, N = self.shape
        if self.dtype is None:
            dt = "unspecified dtype"
        else:
            dt = "dtype=" + str(self.dtype)

        return "<%dx%d %s with %s>" % (M, N, self.__class__.__name__, dt)

    def _copy_attributes(
        self,
        dest: LinearOperator,
        exclude: Optional[List[str]] = None,
    ) -> None:
        """Copy attributes from one LinearOperator to another"""
        if exclude is None:
            exclude = ["name"]
        attrs = ["dims", "dimsd", "clinear", "explicit", "forceflat", "name"]
        if exclude is not None:
            for item in exclude:
                attrs.remove(item)
        for attr in attrs:
            if hasattr(self, attr):
                setattr(dest, attr, getattr(self, attr))

    def _matvec(self, x: NDArray) -> NDArray:
        """Matrix-vector multiplication handler."""
        if self.Op is not None:
            return self.Op._matvec(x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        """Matrix-vector adjoint multiplication handler."""
        if self.Op is not None:
            return self.Op._rmatvec(x)

    def _matmat(self, X: NDArray) -> NDArray:
        """Matrix-matrix multiplication handler.

        Modified version of scipy _matmat to avoid having trailing dimension
        in col when provided to matvec
        """
        if sp.sparse.issparse(X):
            y = np.vstack([self.matvec(col.toarray().reshape(-1)) for col in X.T]).T
        else:
            y = np.vstack([self.matvec(col.reshape(-1)) for col in X.T]).T
        return y

    def _rmatmat(self, X: NDArray) -> NDArray:
        """Matrix-matrix adjoint multiplication handler.

        Modified version of scipy _rmatmat to avoid having trailing dimension
        in col when provided to rmatvec
        """
        if sp.sparse.issparse(X):
            y = np.vstack([self.rmatvec(col.toarray().reshape(-1)) for col in X.T]).T
        else:
            y = np.vstack([self.rmatvec(col.reshape(-1)) for col in X.T]).T
        return y

    def _adjoint(self) -> LinearOperator:
        Op = _AdjointLinearOperator(self)
        self._copy_attributes(Op, exclude=["dims", "dimsd", "explicit", "name"])
        Op.explicit = False
        Op.dims = self.dimsd
        Op.dimsd = self.dims
        return Op

    def _transpose(self) -> LinearOperator:
        Op = _TransposedLinearOperator(self)
        self._copy_attributes(Op, exclude=["dims", "dimsd", "explicit", "name"])
        Op.explicit = False
        Op.dims = self.dimsd
        Op.dimsd = self.dims
        return Op

    def adjoint(self):
        return self._adjoint()

    H: Callable[[LinearOperator], LinearOperator] = property(adjoint)

    def transpose(self):
        return self._transpose()

    T: Callable[[LinearOperator], LinearOperator] = property(transpose)

    @count(forward=True)
    def matvec(self, x: NDArray) -> NDArray:
        """Matrix-vector multiplication.

        Modified version of scipy matvec which does not consider the case
        where the input vector is ``np.matrix`` (the use ``np.matrix`` is now
        discouraged in numpy's documentation).

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Input array of shape (N,) or (N,1)

        Returns
        -------
        y : :obj:`numpy.ndarray`
            Output array of shape (M,) or (M,1)

        """
        M, N = self.shape

        if x.shape != (N,) and x.shape != (N, 1):
            raise ValueError("dimension mismatch")

        y = self._matvec(x)

        if x.ndim == 1:
            y = y.reshape(M)
        elif x.ndim == 2:
            y = y.reshape(M, 1)
        else:
            raise ValueError("invalid shape returned by user-defined matvec()")
        return y

    @count(forward=False)
    def rmatvec(self, x: NDArray) -> NDArray:
        """Adjoint matrix-vector multiplication.

        Modified version of scipy rmatvec which does not consider the case
        where the input vector is ``np.matrix`` (the use ``np.matrix`` is now
        discouraged in numpy's documentation).

        Parameters
        ----------
        y : :obj:`numpy.ndarray`
            Input array of shape (M,) or (M,1)

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Output array of shape (N,) or (N,1)

        """
        M, N = self.shape

        if x.shape != (M,) and x.shape != (M, 1):
            raise ValueError("dimension mismatch")

        y = self._rmatvec(x)

        if x.ndim == 1:
            y = y.reshape(N)
        elif x.ndim == 2:
            y = y.reshape(N, 1)
        else:
            raise ValueError("invalid shape returned by user-defined rmatvec()")
        return y

    @count(forward=True, matmat=True)
    def matmat(self, X: NDArray) -> NDArray:
        """Matrix-matrix multiplication.

        Modified version of scipy matmat which does not consider the case
        where the input vector is ``np.matrix`` (the use ``np.matrix`` is now
        discouraged in numpy's documentation).

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Input array of shape (N,K)

        Returns
        -------
        y : :obj:`numpy.ndarray`
            Output array of shape (M,K)

        """
        if X.ndim != 2:
            raise ValueError("expected 2-d ndarray or matrix, " "not %d-d" % X.ndim)
        if X.shape[0] != self.shape[1]:
            raise ValueError("dimension mismatch: %r, %r" % (self.shape, X.shape))
        Y = self._matmat(X)
        return Y

    @count(forward=False, matmat=True)
    def rmatmat(self, X: NDArray) -> NDArray:
        """Matrix-matrix multiplication.

        Modified version of scipy rmatmat which does not consider the case
        where the input vector is ``np.matrix`` (the use ``np.matrix`` is now
        discouraged in numpy's documentation).

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Input array of shape (M,K)

        Returns
        -------
        y : :obj:`numpy.ndarray`
            Output array of shape (N,K)

        """
        if X.ndim != 2:
            raise ValueError("expected 2-d ndarray or matrix, " "not %d-d" % X.ndim)
        if X.shape[0] != self.shape[0]:
            raise ValueError("dimension mismatch: %r, %r" % (self.shape, X.shape))
        Y = self._rmatmat(X)
        return Y

    def dot(self, x: NDArray) -> NDArray:
        """Matrix-matrix or matrix-vector multiplication.

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Input array (or matrix)

        Returns
        -------
        y : :obj:`numpy.ndarray`
            Output array (or matrix) that represents
            the result of applying the linear operator on x.

        """
        if isinstance(x, (LinearOperator, spLinearOperator)):
            # cast x to pylops linear operator if not already (this is done
            # to allow mixing pylops and scipy operators)
            Opx = aslinearoperator(x)
            Op = _ProductLinearOperator(self, Opx)
            self._copy_attributes(Op, exclude=["dims", "explicit", "forceflat", "name"])
            Op.clinear = Op.clinear and Opx.clinear
            Op.explicit = False
            if self.forceflat is None and Opx.forceflat is None:
                Op.forceflat = None
            elif self.forceflat is not None and Opx.forceflat is not None:
                # Define forceflat only if differing, otherwise raise error
                if self.forceflat != Opx.forceflat:
                    raise ValueError(
                        f"Operators have conflicting forceflat {Op.forceflat} != {Opx.forceflat}"
                    )
                Op.forceflat = self.forceflat
            else:  # Only one of them is None
                Op.forceflat = (
                    self.forceflat if self.forceflat is not None else Opx.forceflat
                )
            Op.dims = Opx.dims
            return Op
        elif np.isscalar(x):
            Op = _ScaledLinearOperator(self, x)
            self._copy_attributes(
                Op,
                exclude=["explicit", "name"],
            )
            Op.explicit = False
            return Op
        else:
            if not get_ndarray_multiplication() and (
                x.ndim > 2 or (x.ndim == 2 and x.shape[0] != self.shape[1])
            ):
                raise ValueError(
                    "Operator can only be applied 1D vectors or 2D matrices. "
                    "Enable ndarray multiplication with pylops.set_ndarray_multiplication(True)."
                )
            is_dims_shaped = x.shape == self.dims
            is_dims_shaped_matrix = len(x.shape) > 1 and x.shape[:-1] == self.dims
            if is_dims_shaped:
                # (dims1, ..., dimsK) => (dims1 * ... * dimsK,) == self.shape
                x = x.ravel()
            if is_dims_shaped_matrix and not self.forceflat:
                # (dims1, ..., dimsK, P) => (dims1 * ... * dimsK, P)
                x = x.reshape((-1, x.shape[-1]))
            if x.ndim == 1:
                y = self.matvec(x)
                if (
                    is_dims_shaped
                    and not self.forceflat
                    and get_ndarray_multiplication()
                ):
                    y = y.reshape(self.dimsd)
                return y
            elif x.ndim == 2:
                y = self.matmat(x)
                if (
                    is_dims_shaped_matrix
                    and not self.forceflat
                    and get_ndarray_multiplication()
                ):
                    y = y.reshape((*self.dimsd, -1))
                return y
            else:
                raise ValueError(
                    (
                        "Wrong shape.\nFor vector multiplication, expects either a 1d "
                        "array or, an ndarray of size `dims` when `dims` and `dimsd` "
                        "both are available.\n"
                        "For matrix multiplication, expects a 2d array with its first "
                        f"dimension is equal to {self.shape[1]}.\n"
                        f"Instead, received an array of shape {x.shape}."
                    )
                )

    def div(
        self,
        y: NDArray,
        niter: int = 100,
        densesolver: str = "scipy",
    ) -> NDArray:
        r"""Solve the linear problem :math:`\mathbf{y}=\mathbf{A}\mathbf{x}`.

        Overloading of operator ``/`` to improve expressivity of `Pylops`
        when solving inverse problems.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data
        niter : :obj:`int`, optional
            Number of iterations (to be used only when ``explicit=False``)
        densesolver : :obj:`str`, optional
            Use scipy (``scipy``) or numpy (``numpy``) dense solver

        Returns
        -------
        xest : :obj:`np.ndarray`
            Estimated model

        """
        xest = self.__truediv__(y, niter=niter, densesolver=densesolver)
        return xest

    def __truediv__(
        self,
        y: NDArray,
        niter: int = 100,
        densesolver: str = "scipy",
    ) -> NDArray:
        if self.explicit is True:
            if sp.sparse.issparse(self.A):
                # use scipy solver for sparse matrices
                xest = spsolve(self.A, y)
            elif isinstance(self.A, np.ndarray):
                # use scipy solvers for dense matrices (used for backward
                # compatibility, could be switched to numpy equivalents)
                if self.A.shape[0] == self.A.shape[1]:
                    if densesolver == "scipy":
                        xest = sp_solve(self.A, y)
                    else:
                        xest = np_solve(self.A, y)
                else:
                    xest = lstsq(self.A, y)[0]
            else:
                # use numpy/cupy solvers for dense matrices
                ncp = get_array_module(y)
                if self.A.shape[0] == self.A.shape[1]:
                    xest = ncp.linalg.solve(self.A, y)
                else:
                    xest = ncp.linalg.lstsq(self.A, y)[0]
        else:
            if isinstance(y, np.ndarray):
                # numpy backend
                xest = lsqr(self, y, iter_lim=niter, atol=1e-8, btol=1e-8)[0]
            else:
                # cupy backend
                ncp = get_array_module(y)
                xest = cgls(
                    self,
                    y,
                    x0=ncp.zeros(int(self.shape[1]), dtype=self.dtype),
                    niter=niter,
                )[0]
        return xest

    def todense(
        self,
        backend: str = "numpy",
    ) -> NDArray:
        r"""Return dense matrix.

        The operator is converted into its dense matrix equivalent. In order
        to do so, square or tall operators are applied to an identity matrix
        whose number of rows and columns is equivalent to the number of
        columns of the operator. Conversely, for skinny operators, the
        transpose operator is applied to an identity matrix
        whose number of rows and columns is equivalent to the number of
        rows of the operator and the resulting matrix is transposed
        (and complex conjugated).

        Note that this operation may be costly for operators with large number
        of rows and columns and it should be used mostly as a way to inspect
        the structure of the matricial equivalent of the operator.

        Parameters
        ----------
        backend : :obj:`str`, optional
            Backend used to densify matrix (``numpy`` or ``cupy``). Note that
            this must be consistent with how the operator has been created.

        Returns
        -------
        matrix : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
            Dense matrix.

        """
        ncp = get_module(backend)
        Op = self

        # Create identity matrix
        shapemin = min(Op.shape)
        if shapemin <= 1e3:
            # use numpy for small matrices (faster but heavier on memory)
            identity = ncp.eye(shapemin, dtype=self.dtype)
        else:
            # use scipy for large matrices (slower but lighter on memory)
            identity = get_sparse_eye(ncp.ones(1))(shapemin, dtype=self.dtype).tocsc()

        # Apply operator
        if Op.shape[1] == shapemin:
            matrix = Op.matmat(identity)
        else:
            matrix = np.conj(Op.rmatmat(identity)).T
        return matrix

    def tosparse(self) -> NDArray:
        r"""Return sparse matrix.

        The operator in converted into its sparse (CSR) matrix equivalent. In order
        to do so, the operator is applied to series of unit vectors with length equal
        to the number of coloumns in the original operator.

        Returns
        -------
        matrix : :obj:`scipy.sparse.csr_matrix`
            Sparse matrix.

        """
        Op = self
        (_, n) = self.shape

        # stores non-zero data for the sparse matrix creation
        entries = []
        indices = []

        # loop through columns of self
        for i in range(n):
            # make a unit vector for the ith column
            unit_i = np.zeros(n)
            unit_i[i] = 1

            # multiply unit vector to self and find the non-zeros
            res_i: NDArray = Op * unit_i
            rows_nz = np.where(res_i != 0)[0]

            # append the non-zero values and indices to the lists
            for j in rows_nz:
                indices.append([i, j])
            entries_i = res_i[rows_nz]
            for e in entries_i:
                entries.append(e)

        # post process the entries / indices for scipy.sparse.csr_matrix
        entries = np.array(entries)
        indices = np.array(indices)
        i, j = indices[:, 0], indices[:, 1]

        # construct a sparse, CSR matrix from the entries / indices data.
        matrix = csr_matrix((entries, (j, i)), shape=self.shape, dtype=self.dtype)
        return matrix

    def eigs(
        self,
        neigs: Optional[int] = None,
        symmetric: bool = False,
        niter: Optional[int] = None,
        uselobpcg: bool = False,
        **kwargs_eig: Union[int, float, str],
    ) -> NDArray:
        r"""Most significant eigenvalues of linear operator.

        Return an estimate of the most significant eigenvalues
        of the linear operator. If the operator has rectangular
        shape (``shape[0]!=shape[1]``), eigenvalues are first
        computed for the square operator :math:`\mathbf{A^H}\mathbf{A}`
        and the square-root values are returned.

        Parameters
        ----------
        neigs : :obj:`int`
            Number of eigenvalues to compute (if ``None``, return all). Note
            that for ``explicit=False``, only :math:`N-1` eigenvalues can be
            computed where :math:`N` is the size of the operator in the
            model space
        symmetric : :obj:`bool`, optional
            Operator is symmetric (``True``) or not (``False``). User should
            set this parameter to ``True`` only when it is guaranteed that the
            operator is real-symmetric or complex-hermitian matrices
        niter : :obj:`int`, optional
            Number of iterations for eigenvalue estimation
        uselobpcg : :obj:`bool`, optional
            Use :func:`scipy.sparse.linalg.lobpcg`
        **kwargs_eig
            Arbitrary keyword arguments for :func:`scipy.sparse.linalg.eigs`,
            :func:`scipy.sparse.linalg.eigsh`, or
            :func:`scipy.sparse.linalg.lobpcg`

        Returns
        -------
        eigenvalues : :obj:`numpy.ndarray`
            Operator eigenvalues.

        Raises
        ------
        ValueError
            If ``uselobpcg=True`` for a non-symmetric square matrix with
            complex type

        Notes
        -----
        Depending on the size of the operator, whether it is explicit or not
        and the number of eigenvalues requested, different algorithms are
        used by this routine.

        More precisely, when only a limited number of eigenvalues is requested
        the :func:`scipy.sparse.linalg.eigsh` method is used in case of
        ``symmetric=True`` and the :func:`scipy.sparse.linalg.eigs` method
        is used ``symmetric=False``. However, when the matrix is represented
        explicitly within the linear operator (``explicit=True``) and all the
        eigenvalues are requested the :func:`scipy.linalg.eigvals` is used
        instead.

        Finally, when only a limited number of eigenvalues is required,
        it is also possible to explicitly choose to use the
        ``scipy.sparse.linalg.lobpcg`` method via the ``uselobpcg`` input
        parameter flag.

        Most of these algorithms are a port of ARPACK [1]_, a Fortran package
        which provides routines for quickly finding eigenvalues/eigenvectors
        of a matrix. As ARPACK requires only left-multiplication
        by the matrix in question, eigenvalues/eigenvectors can also be
        estimated for linear operators when the dense matrix is not available.

        .. [1] http://www.caam.rice.edu/software/ARPACK/

        """
        if self.explicit and isinstance(self.A, np.ndarray):
            if self.shape[0] == self.shape[1]:
                if neigs is None or neigs == self.shape[1]:
                    eigenvalues = eigvals(self.A)
                else:
                    if not symmetric and np.iscomplexobj(self) and uselobpcg:
                        raise ValueError(
                            "cannot use scipy.sparse.linalg.lobpcg "
                            "for non-symmetric square matrices of "
                            "complex type..."
                        )
                    if symmetric and uselobpcg:
                        X = np.random.rand(self.shape[0], neigs).astype(self.dtype)
                        eigenvalues = sp_lobpcg(
                            self.A, X=X, maxiter=niter, **kwargs_eig
                        )[0]
                    elif symmetric:
                        eigenvalues = sp_eigsh(
                            self.A, k=neigs, maxiter=niter, **kwargs_eig
                        )[0]
                    else:
                        eigenvalues = sp_eigs(
                            self.A, k=neigs, maxiter=niter, **kwargs_eig
                        )[0]

            else:
                if neigs is None or neigs == self.shape[1]:
                    eigenvalues = np.sqrt(eigvals(np.dot(np.conj(self.A.T), self.A)))
                else:
                    if uselobpcg:
                        X = np.random.rand(self.shape[1], neigs).astype(self.dtype)
                        eigenvalues = np.sqrt(
                            sp_lobpcg(
                                np.dot(np.conj(self.A.T), self.A),
                                X=X,
                                maxiter=niter,
                                **kwargs_eig,
                            )[0]
                        )
                    else:
                        eigenvalues = np.sqrt(
                            sp_eigsh(
                                np.dot(np.conj(self.A.T), self.A),
                                k=neigs,
                                maxiter=niter,
                                **kwargs_eig,
                            )[0]
                        )
        else:
            if neigs is None or neigs >= self.shape[1]:
                neigs = self.shape[1] - 2
            if self.shape[0] == self.shape[1]:
                if not symmetric and np.iscomplexobj(self) and uselobpcg:
                    raise ValueError(
                        "cannot use scipy.sparse.linalg.lobpcg for "
                        "non symmetric square matrices of "
                        "complex type..."
                    )
                if symmetric and uselobpcg:
                    X = np.random.rand(self.shape[0], neigs).astype(self.dtype)
                    eigenvalues = sp_lobpcg(self, X=X, maxiter=niter, **kwargs_eig)[0]
                elif symmetric:
                    eigenvalues = sp_eigsh(self, k=neigs, maxiter=niter, **kwargs_eig)[
                        0
                    ]
                else:
                    eigenvalues = sp_eigs(self, k=neigs, maxiter=niter, **kwargs_eig)[0]
            else:
                if uselobpcg:
                    X = np.random.rand(self.shape[1], neigs).astype(self.dtype)
                    eigenvalues = np.sqrt(
                        sp_lobpcg(self.H * self, X=X, maxiter=niter, **kwargs_eig)[0]
                    )
                else:
                    eigenvalues = np.sqrt(
                        sp_eigs(self.H * self, k=neigs, maxiter=niter, **kwargs_eig)[0]
                    )
        return -np.sort(-eigenvalues)

    def cond(
        self,
        uselobpcg: bool = False,
        **kwargs_eig: Union[int, float, str],
    ) -> NDArray:
        r"""Condition number of linear operator.

        Return an estimate of the condition number of the linear operator as
        the ratio of the largest and lowest estimated eigenvalues.

        Parameters
        ----------
        uselobpcg : :obj:`bool`, optional
            Use :func:`scipy.sparse.linalg.lobpcg` to compute eigenvalues
        **kwargs_eig
            Arbitrary keyword arguments for :func:`scipy.sparse.linalg.eigs`,
            :func:`scipy.sparse.linalg.eigsh`, or
            :func:`scipy.sparse.linalg.lobpcg`

        Returns
        -------
        eigenvalues : :obj:`numpy.ndarray`
            Operator eigenvalues.

        Notes
        -----
        The condition number of a matrix (or linear operator) can be estimated
        as the ratio of the largest and lowest estimated eigenvalues:

        .. math::
            k= \frac{\lambda_{max}}{\lambda_{min}}

        The condition number provides an indication of the rate at which the
        solution of the inversion of the linear operator :math:`A` will
        change with respect to a change in the data :math:`y`.

        Thus, if the condition number is large, even a small error in :math:`y`
        may cause a large error in :math:`x`. On the other hand, if the
        condition number is small then the error in :math:`x` is not much
        bigger than the error in :math:`y`. A problem with a low condition
        number is said to be *well-conditioned*, while a problem with a high
        condition number is said to be *ill-conditioned*.

        """
        if not uselobpcg:
            cond = (
                self.eigs(neigs=1, which="LM", **kwargs_eig).item()
                / self.eigs(neigs=1, which="SM", **kwargs_eig).item()
            )
        else:
            cond = (
                self.eigs(neigs=1, uselobpcg=True, largest=True, **kwargs_eig).item()
                / self.eigs(neigs=1, uselobpcg=True, largest=False, **kwargs_eig).item()
            )

        return cond

    def conj(self) -> LinearOperator:
        """Complex conjugate operator

        Returns
        -------
        conjop : :obj:`pylops.LinearOperator`
            Complex conjugate operator

        """
        conjop = _ConjLinearOperator(self)
        return conjop

    def apply_columns(self, cols: InputDimsLike) -> LinearOperator:
        """Apply subset of columns of operator

        This method can be used to wrap a LinearOperator and mimic the action
        of a subset of columns of the operator on a reduced model in forward
        mode, and retrieve only the result of a subset of rows in adjoint mode.

        Note that unless the operator has ``explicit=True``, this is not
        optimal as the entire forward and adjoint passes of the original
        operator will have to be perfomed. It can however be useful for the
        implementation of solvers such as Orthogonal Matching Pursuit (OMP)
        that iteratively build a solution by evaluate only a subset of the
        columns of the operator.

        Parameters
        ----------
        cols : :obj:`list`
            Columns to be selected

        Returns
        -------
        colop : :obj:`pylops.LinearOperator`
            Apply column operator

        """
        colop = _ColumnLinearOperator(self, cols)
        return colop

    def toreal(self, forw: bool = True, adj: bool = True) -> LinearOperator:
        """Real operator

        Parameters
        ----------
        forw : :obj:`bool`, optional
            Apply real to output of forward pass
        adj : :obj:`bool`, optional
            Apply real to output of adjoint pass

        Returns
        -------
        realop : :obj:`pylops.LinearOperator`
            Real operator

        """
        realop = _RealImagLinearOperator(self, forw, adj, True)
        return realop

    def toimag(self, forw: bool = True, adj: bool = True) -> LinearOperator:
        """Imag operator

        Parameters
        ----------
        forw : :obj:`bool`, optional
            Apply imag to output of forward pass
        adj : :obj:`bool`, optional
            Apply imag to output of adjoint pass

        Returns
        -------
        imagop : :obj:`pylops.LinearOperator`
            Imag operator

        """
        imagop = _RealImagLinearOperator(self, forw, adj, False)
        return imagop

    def trace(
        self,
        neval: Optional[int] = None,
        method: Optional[str] = None,
        backend: str = "numpy",
        **kwargs_trace,
    ) -> float:
        r"""Trace of linear operator.

        Returns the trace (or its estimate) of the linear operator.

        Parameters
        ----------
        neval : :obj:`int`, optional
            Maximum number of matrix-vector products compute. Default depends
            ``method``.
        method : :obj:`str`, optional
            Should be one of the following:

                - **explicit**: If the operator is not explicit, will convert to
                  dense first.
                - **hutchinson**: see :obj:`pylops.utils.trace_hutchinson`
                - **hutch++**: see :obj:`pylops.utils.trace_hutchpp`
                - **na-hutch++**: see :obj:`pylops.utils.trace_nahutchpp`

            Defaults to 'explicit' for explicit operators, and 'Hutch++' for
            the rest.
        backend : :obj:`str`, optional
            Backend used to densify matrix (``numpy`` or ``cupy``). Note that
            this must be consistent with how the operator has been created.
        **kwargs_trace
            Arbitrary keyword arguments passed to
            :obj:`pylops.utils.trace_hutchinson`,
            :obj:`pylops.utils.trace_hutchpp`, or
            :obj:`pylops.utils.trace_nahutchpp`

        Returns
        -------
        trace : :obj:`float`
            Operator trace.

        Raises
        ------
        ValueError
             If the operator has rectangular shape (``shape[0] != shape[1]``)

        NotImplementedError
            If the ``method`` is not one of the available methods.
        """
        if self.shape[0] != self.shape[1]:
            raise ValueError("operator is not square.")

        ncp = get_module(backend)

        if method is None:
            method = "explicit" if self.explicit else "hutch++"

        method_l = method.lower()
        if method_l == "explicit":
            A = self.A if self.explicit else self.todense(backend=backend)
            return ncp.trace(A)
        elif method_l == "hutchinson":
            return trace_hutchinson(self, neval=neval, backend=backend, **kwargs_trace)
        elif method_l == "hutch++":
            return trace_hutchpp(self, neval=neval, backend=backend, **kwargs_trace)
        elif method_l == "na-hutch++":
            return trace_nahutchpp(self, neval=neval, backend=backend, **kwargs_trace)
        else:
            raise NotImplementedError(f"method {method} not available.")

    def reset_count(self) -> None:
        """Reset counters

        When invoked all counters are set back to 0.

        """
        self.matvec_count = 0
        self.rmatvec_count = 0
        self.matmat_count = 0
        self.rmatmat_count = 0


def _get_dtype(
    operators: Sequence[LinearOperator],
    dtypes: Optional[Sequence[DTypeLike]] = None,
) -> DTypeLike:
    if dtypes is None:
        dtypes = []
    for obj in operators:
        if obj is not None and hasattr(obj, "dtype"):
            dtypes.append(obj.dtype)
    return np.result_type(*dtypes)


class _ScaledLinearOperator(LinearOperator):
    """Scaled Linear Operator"""

    def __init__(
        self,
        A: LinearOperator,
        alpha: float,
    ) -> None:
        if not isinstance(A, LinearOperator):
            raise ValueError("LinearOperator expected as A")
        if not np.isscalar(alpha):
            raise ValueError("scalar expected as alpha")
        if isinstance(alpha, complex) and not np.iscomplexobj(
            np.ones(1, dtype=A.dtype)
        ):
            # if the scalar is of complex type but not the operator, find out type
            dtype = _get_dtype([A], [type(alpha)])
        else:
            # if both the scalar and operator are of real or complex type, use type
            # of the operator
            dtype = A.dtype
        super(_ScaledLinearOperator, self).__init__(dtype=dtype, shape=A.shape)
        self.args = (A, alpha)

    def _matvec(self, x: NDArray) -> NDArray:
        return self.args[1] * self.args[0].matvec(x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        return np.conj(self.args[1]) * self.args[0].rmatvec(x)

    def _rmatmat(self, x: NDArray) -> NDArray:
        return np.conj(self.args[1]) * self.args[0].rmatmat(x)

    def _matmat(self, x: NDArray) -> NDArray:
        return self.args[1] * self.args[0].matmat(x)

    def _adjoint(self) -> LinearOperator:
        A, alpha = self.args
        return A.H * np.conj(alpha)


class _ConjLinearOperator(LinearOperator):
    """Complex conjugate linear operator"""

    def __init__(self, Op: LinearOperator) -> None:
        if not isinstance(Op, LinearOperator):
            raise TypeError("Op must be a LinearOperator")
        super(_ConjLinearOperator, self).__init__(Op, shape=Op.shape)
        self.Op = Op

    def _matvec(self, x: NDArray) -> NDArray:
        return (self.Op._matvec(x.conj())).conj()

    def _rmatvec(self, x: NDArray) -> NDArray:
        return (self.Op._rmatvec(x.conj())).conj()

    def _adjoint(self) -> LinearOperator:
        return _ConjLinearOperator(self.Op.H)


class _ColumnLinearOperator(LinearOperator):
    """Column selector linear operator

    Produces the forward and adjoint passes with a subset of columns of the
    original operator
    """

    def __init__(
        self,
        Op: LinearOperator,
        cols: InputDimsLike,
    ) -> None:
        if not isinstance(Op, LinearOperator):
            raise TypeError("Op must be a LinearOperator")
        super(_ColumnLinearOperator, self).__init__(Op, explicit=Op.explicit)
        self.Op = Op
        self.cols = cols
        self._shape = (Op.shape[0], len(cols))
        self._dims = len(cols)
        if self.explicit:
            self.Opcol = Op.A[:, cols]

    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if self.explicit:
            y = self.Opcol @ x
        else:
            y = ncp.zeros(int(self.Op.shape[1]), dtype=self.dtype)
            y[self.cols] = x
            y = self.Op._matvec(y)
        return y

    def _rmatvec(self, x: NDArray) -> NDArray:
        if self.explicit:
            y = self.Opcol.T.conj() @ x
        else:
            y = self.Op._rmatvec(x)
            y = y[self.cols]
        return y


class _AdjointLinearOperator(LinearOperator):
    """Adjoint of Linear Operator"""

    def __init__(self, A: LinearOperator):
        shape = (A.shape[1], A.shape[0])
        super(_AdjointLinearOperator, self).__init__(shape=shape, dtype=A.dtype)
        self.A = A
        self.args = (A,)

    def _matvec(self, x: NDArray) -> NDArray:
        return self.A._rmatvec(x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        return self.A._matvec(x)

    def _matmat(self, X: NDArray) -> NDArray:
        return self.A._rmatmat(X)

    def _rmatmat(self, X: NDArray) -> NDArray:
        return self.A._matmat(X)


class _TransposedLinearOperator(LinearOperator):
    """Transposition of Linear Operator"""

    def __init__(self, A: LinearOperator):
        shape = (A.shape[1], A.shape[0])
        super(_TransposedLinearOperator, self).__init__(shape=shape, dtype=A.dtype)
        self.A = A
        self.args = (A,)

    def _matvec(self, x: NDArray) -> NDArray:
        return np.conj(self.A._rmatvec(np.conj(x)))

    def _rmatvec(self, x: NDArray) -> NDArray:
        return np.conj(self.A._matvec(np.conj(x)))

    def _matmat(self, X: NDArray) -> NDArray:
        return np.conj(self.A._rmatmat(np.conj(X)))

    def _rmatmat(self, X: NDArray) -> NDArray:
        return np.conj(self.A._matmat(np.conj(X)))


class _ProductLinearOperator(LinearOperator):
    """Product of Linear Operators"""

    def __init__(self, A: LinearOperator, B: LinearOperator):
        if not isinstance(A, LinearOperator) or not isinstance(B, LinearOperator):
            raise ValueError(
                f"both operands have to be a LinearOperator{type(A)} {type(B)}"
            )
        if A.shape[1] != B.shape[0]:
            raise ValueError("cannot add %r and %r: shape mismatch" % (A, B))
        super().__init__(dtype=_get_dtype([A, B]), shape=(A.shape[0], B.shape[1]))
        self.args = (A, B)

    def _matvec(self, x: NDArray) -> NDArray:
        return self.args[0].matvec(self.args[1].matvec(x))

    def _rmatvec(self, x: NDArray) -> NDArray:
        return self.args[1].rmatvec(self.args[0].rmatvec(x))

    def _rmatmat(self, X: NDArray) -> NDArray:
        return self.args[1].rmatmat(self.args[0].rmatmat(X))

    def _matmat(self, X: NDArray) -> NDArray:
        return self.args[0].matmat(self.args[1].matmat(X))

    def _adjoint(self):
        A, B = self.args
        return B.H * A.H


class _SumLinearOperator(LinearOperator):
    def __init__(
        self,
        A: LinearOperator,
        B: LinearOperator,
    ) -> None:
        if not isinstance(A, LinearOperator) or not isinstance(B, LinearOperator):
            raise ValueError("both operands have to be a LinearOperator")
        if A.shape != B.shape:
            raise ValueError("cannot add %r and %r: shape mismatch" % (A, B))
        self.args = (A, B)
        super(_SumLinearOperator, self).__init__(
            dtype=_get_dtype([A, B]), shape=A.shape
        )

    def _matvec(self, x: NDArray) -> NDArray:
        return self.args[0].matvec(x) + self.args[1].matvec(x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        return self.args[0].rmatvec(x) + self.args[1].rmatvec(x)

    def _rmatmat(self, x: NDArray) -> NDArray:
        return self.args[0].rmatmat(x) + self.args[1].rmatmat(x)

    def _matmat(self, x: NDArray) -> NDArray:
        return self.args[0].matmat(x) + self.args[1].matmat(x)

    def _adjoint(self) -> LinearOperator:
        A, B = self.args
        return A.H + B.H


class _PowerLinearOperator(LinearOperator):
    def __init__(self, A: LinearOperator, p: int) -> None:
        if not isinstance(A, LinearOperator):
            raise ValueError("LinearOperator expected as A")
        if A.shape[0] != A.shape[1]:
            raise ValueError("square LinearOperator expected, got %r" % A)
        if not isintlike(p) or p < 0:
            raise ValueError("non-negative integer expected as p")

        super(_PowerLinearOperator, self).__init__(dtype=A.dtype, shape=A.shape)
        self.args = (A, p)

    def _power(self, fun: Callable, x: NDArray) -> NDArray:
        res = np.array(x, copy=True)
        for _ in range(self.args[1]):
            res = fun(res)
        return res

    def _matvec(self, x: NDArray) -> NDArray:
        return self._power(self.args[0].matvec, x)

    def _rmatvec(self, x: NDArray) -> NDArray:
        return self._power(self.args[0].rmatvec, x)

    def _rmatmat(self, x: NDArray) -> NDArray:
        return self._power(self.args[0].rmatmat, x)

    def _matmat(self, x: NDArray) -> NDArray:
        return self._power(self.args[0].matmat, x)

    def _adjoint(self) -> LinearOperator:
        A, p = self.args
        return A.H**p


class _RealImagLinearOperator(LinearOperator):
    """Real-Imag linear operator

    Computes forward and adjoint passes of an operator Op and returns only
    its real (or imaginary) component. Note that for the adjoint step the
    output must be complex conjugated (i.e. opposite of the imaginary part is
    returned)
    """

    def __init__(
        self,
        Op: LinearOperator,
        forw: bool = True,
        adj: bool = True,
        real: bool = True,
    ) -> None:
        if not isinstance(Op, LinearOperator):
            raise TypeError("Op must be a LinearOperator")
        super(_RealImagLinearOperator, self).__init__(Op, shape=Op.shape)
        self.Op = Op
        self.real = real
        self.forw = forw
        self.adj = adj
        self.dtype = np.array(0, dtype=self.Op.dtype).real.dtype

    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = self.Op._matvec(x)
        if self.forw:
            if self.real:
                y = ncp.real(y)
            else:
                y = ncp.imag(y)
        return y

    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = self.Op._rmatvec(x)
        if self.adj:
            if self.real:
                y = ncp.real(y)
            else:
                y = -ncp.imag(y)
        return y


def aslinearoperator(Op: Union[spLinearOperator, LinearOperator]) -> LinearOperator:
    """Return Op as a LinearOperator.

    Converts any operator compatible with pylops definition of LinearOperator into a pylops
    LinearOperator. This can be used for example when `Op` is a scipy operator to ensure
    that the returned operator has all properties and methods of the pylops class.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator` or any other Operator
        Operator of any type

    Returns
    -------
    Op : :obj:`pylops.LinearOperator`
        Operator of type :obj:`pylops.LinearOperator`

    """
    if isinstance(Op, LinearOperator):
        return Op
    else:
        return LinearOperator(Op)
