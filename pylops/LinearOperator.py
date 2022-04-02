from __future__ import division

import logging

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
    from scipy.sparse.linalg.interface import _ProductLinearOperator
else:
    from scipy.sparse.linalg._interface import _ProductLinearOperator

from pylops.optimization.solver import cgls
from pylops.utils.backend import get_array_module, get_module, get_sparse_eye
from pylops.utils.estimators import trace_hutchinson, trace_hutchpp, trace_nahutchpp

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class LinearOperator(spLinearOperator):
    """Common interface for performing matrix-vector products.

    This class is an overload of the
    :py:class:`scipy.sparse.linalg.LinearOperator` class. It adds
    functionalities by overloading standard operators such as ``__truediv__``
    as well as creating convenience methods such as ``eigs``, ``cond``, and
    ``conj``.

    .. note:: End users of PyLops should not use this class directly but simply
      use operators that are already implemented. This class is meant for
      developers and it has to be used as the parent class of any new operator
      developed within PyLops. Find more details regarding implementation of
      new operators at :ref:`addingoperator`.

    Parameters
    ----------
    Op : :obj:`scipy.sparse.linalg.LinearOperator` or :obj:`scipy.sparse.linalg._ProductLinearOperator` or :obj:`scipy.sparse.linalg._SumLinearOperator`
        Operator
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)
    clinear : :obj:`bool`
        .. versionadded:: 1.17.0

        Operator is complex-linear.
    """

    def __init__(self, Op=None, explicit=False, clinear=None):
        self.explicit = explicit
        if Op is not None:
            self.Op = Op
            self.shape = self.Op.shape
            self.dtype = self.Op.dtype
            self.clinear = getattr(self.Op, "clinear", True)
        if clinear is not None:
            self.clinear = clinear

    def _matvec(self, x):
        if callable(self.Op._matvec):
            return self.Op._matvec(x)

    def _rmatvec(self, x):
        if callable(self.Op._rmatvec):
            return self.Op._rmatvec(x)

    def _matmat(self, X):
        """Matrix-matrix multiplication handler.

        Modified version of scipy _matmat to avoid having trailing dimension
        in col when provided to matvec
        """
        if sp.sparse.issparse(X):
            y = np.vstack([self.matvec(col.toarray().reshape(-1)) for col in X.T]).T
        else:
            y = np.vstack([self.matvec(col.reshape(-1)) for col in X.T]).T
        return y

    def __mul__(self, x):
        y = super().__mul__(x)
        if isinstance(y, spLinearOperator):
            y = aslinearoperator(y)
        return y

    def __rmul__(self, x):
        if np.isscalar(x):
            return aslinearoperator(_ScaledLinearOperator(self, x))
        else:
            return NotImplemented

    def __pow__(self, p):
        return aslinearoperator(super().__pow__(p))

    def __add__(self, x):
        return aslinearoperator(super().__add__(x))

    def __neg__(self):
        return aslinearoperator(_ScaledLinearOperator(self, -1))

    def __sub__(self, x):
        return aslinearoperator(super().__sub__(x))

    def _adjoint(self):
        return aslinearoperator(super()._adjoint())

    def _transpose(self):
        return aslinearoperator(super()._transpose())

    def matvec(self, x):
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

    def rmatvec(self, x):
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

    def matmat(self, X):
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

    def rmatmat(self, X):
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

    def dot(self, x):
        """Matrix-matrix or matrix-vector multiplication.

        Parameters
        ----------
        x : np.ndarray
            Input array (or matrix)

        Returns
        -------
        y : np.ndarray
            Output array (or matrix) that represents
            the result of applying the linear operator on x.

        """
        if isinstance(x, LinearOperator):
            Op = _ProductLinearOperator(self, x)
            # Output is C-Linear only if both operators are
            Op.clinear = getattr(self, "clinear", True) and getattr(x, "clinear", True)
            return Op
        elif np.isscalar(x):
            return _ScaledLinearOperator(self, x)
        else:
            if x.ndim == 1:  # or x.ndim == 2 and x.shape[1] == 1:
                return self.matvec(x)
            elif x.ndim == 2:
                return self.matmat(x)
            else:
                raise ValueError("expected 1-d or 2-d array or matrix, got %r" % x)

    def div(self, y, niter=100, densesolver="scipy"):
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

    def __truediv__(self, y, niter=100, densesolver="scipy"):
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

    def todense(self, backend="numpy"):
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

        # Wrap self into a LinearOperator. This is done for cases where self
        # is a _SumLinearOperator or _ProductLinearOperator, so that it regains
        # the dense method
        Op = aslinearoperator(self)

        # Create identity matrix
        shapemin = min(Op.shape)
        if shapemin <= 1e3:
            # use numpy for small matrices (faster but heavier on memory)
            identity = ncp.eye(shapemin, dtype=self.dtype)
        else:
            # use scipy for small matrices (slower but lighter on memory)
            identity = get_sparse_eye(ncp.ones(1))(shapemin, dtype=self.dtype).tocsc()

        # Apply operator
        if Op.shape[1] == shapemin:
            matrix = Op.matmat(identity)
        else:
            matrix = np.conj(Op.rmatmat(identity)).T
        return matrix

    def tosparse(self):
        r"""Return sparse matrix.

        The operator in converted into its sparse (CSR) matrix equivalent. In order
        to do so, the operator is applied to series of unit vectors with length equal
        to the number of coloumns in the original operator.

        Returns
        -------
        matrix : :obj:`scipy.sparse.csr_matrix`
            Sparse matrix.

        """
        Op = aslinearoperator(self)
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
            res_i = Op * unit_i
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
        self, neigs=None, symmetric=False, niter=None, uselobpcg=False, **kwargs_eig
    ):
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
        -------
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

    def cond(self, uselobpcg=False, **kwargs_eig):
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

    def conj(self):
        """Complex conjugate operator

        Returns
        -------
        conjop : :obj:`pylops.LinearOperator`
            Complex conjugate operator

        """
        conjop = _ConjLinearOperator(self)
        return conjop

    def apply_columns(self, cols):
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

    def toreal(self, forw=True, adj=True):
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

    def toimag(self, forw=True, adj=True):
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
        neval=None,
        method=None,
        backend="numpy",
        **kwargs_trace,
    ):
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
        trace : :obj:`self.dtype`
            Operator trace.

        Raises
        -------
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


def _get_dtype(operators, dtypes=None):
    if dtypes is None:
        dtypes = []
    opdtypes = []
    for obj in operators:
        if obj is not None and hasattr(obj, "dtype"):
            opdtypes.append(obj.dtype)
    return np.find_common_type(opdtypes, dtypes)


class _ScaledLinearOperator(spLinearOperator):
    """
    Sum Linear Operator

    Modified version of scipy _ScaledLinearOperator which uses a modified
    _get_dtype where the scalar and operator types are passed separately to
    np.find_common_type. Passing them together does lead to problems when using
    np.float32 operators which are cast to np.float64

    """

    def __init__(self, A, alpha):
        if not isinstance(A, spLinearOperator):
            raise ValueError("LinearOperator expected as A")
        if not np.isscalar(alpha):
            raise ValueError("scalar expected as alpha")
        dtype = _get_dtype([A], [type(alpha)])
        super(_ScaledLinearOperator, self).__init__(dtype, A.shape)
        self.args = (A, alpha)

    def _matvec(self, x):
        return self.args[1] * self.args[0].matvec(x)

    def _rmatvec(self, x):
        return np.conj(self.args[1]) * self.args[0].rmatvec(x)

    def _rmatmat(self, x):
        return np.conj(self.args[1]) * self.args[0].rmatmat(x)

    def _matmat(self, x):
        return self.args[1] * self.args[0].matmat(x)

    def _adjoint(self):
        A, alpha = self.args
        return A.H * np.conj(alpha)


class _ConjLinearOperator(LinearOperator):
    """Complex conjugate linear operator"""

    def __init__(self, Op):
        if not isinstance(Op, spLinearOperator):
            raise TypeError("Op must be a LinearOperator")
        super(_ConjLinearOperator, self).__init__(Op, Op.shape)
        self.Op = Op

    def _matvec(self, x):
        return (self.Op._matvec(x.conj())).conj()

    def _rmatvec(self, x):
        return (self.Op._rmatvec(x.conj())).conj()

    def _adjoint(self):
        return _ConjLinearOperator(self.Op.H)


class _ColumnLinearOperator(LinearOperator):
    """Column selector linear operator

    Produces the forward and adjoint passes with a subset of columns of the
    original operator
    """

    def __init__(self, Op, cols):
        if not isinstance(Op, spLinearOperator):
            raise TypeError("Op must be a LinearOperator")
        super(_ColumnLinearOperator, self).__init__(Op, Op.explicit)
        self.Op = Op
        self.cols = cols
        self.shape = (Op.shape[0], len(cols))
        if self.explicit:
            self.Opcol = Op.A[:, cols]

    def _matvec(self, x):
        ncp = get_array_module(x)
        if self.explicit:
            y = self.Opcol @ x
        else:
            y = ncp.zeros(int(self.Op.shape[1]), dtype=self.dtype)
            y[self.cols] = x
            y = self.Op._matvec(y)
        return y

    def _rmatvec(self, x):
        if self.explicit:
            y = self.Opcol.T.conj() @ x
        else:
            y = self.Op._rmatvec(x)
            y = y[self.cols]
        return y


class _RealImagLinearOperator(LinearOperator):
    """Real-Imag linear operator

    Computes forward and adjoint passes of an operator Op and returns only
    its real (or imaginary) component. Note that for the adjoint step the
    output must be complex conjugated (i.e. opposite of the imaginary part is
    returned)
    """

    def __init__(self, Op, forw=True, adj=True, real=True):
        if not isinstance(Op, spLinearOperator):
            raise TypeError("Op must be a LinearOperator")
        super(_RealImagLinearOperator, self).__init__(Op, Op.shape)
        self.Op = Op
        self.real = real
        self.forw = forw
        self.adj = adj
        self.dtype = np.array(0, dtype=self.Op.dtype).real.dtype

    def _matvec(self, x):
        ncp = get_array_module(x)
        y = self.Op._matvec(x)
        if self.forw:
            if self.real:
                y = ncp.real(y)
            else:
                y = ncp.imag(y)
        return y

    def _rmatvec(self, x):
        ncp = get_array_module(x)
        y = self.Op._rmatvec(x)
        if self.adj:
            if self.real:
                y = ncp.real(y)
            else:
                y = -ncp.imag(y)
        return y


def aslinearoperator(Op):
    """Return Op as a LinearOperator.

    Converts any operator into a LinearOperator. This can be used when `Op`
    is a private operator to ensure that the return operator has all properties
    and methods of the parent class.

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
