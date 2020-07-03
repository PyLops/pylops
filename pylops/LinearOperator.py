from __future__ import division

import logging
import numpy as np
from scipy.linalg import solve, lstsq
from scipy.sparse.linalg import LinearOperator as spLinearOperator
from scipy.sparse.linalg import spsolve, lsqr
from scipy.linalg import eigvals
from scipy.sparse.linalg import eigs as sp_eigs
from scipy.sparse.linalg import eigsh as sp_eigsh
from scipy.sparse.linalg import lobpcg as sp_lobpcg
from scipy.sparse import csr_matrix

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


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

    """
    def __init__(self, Op=None, explicit=False):
        self.explicit = explicit
        if Op is not None:
            self.Op = Op
            self.shape = self.Op.shape
            self.dtype = self.Op.dtype

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
        return np.vstack([self.matvec(col.reshape(-1)) for col in X.T]).T

    def __mul__(self, x):
        y = super().__mul__(x)
        if isinstance(y, spLinearOperator):
            y = aslinearoperator(y)
        return y

    def __rmul__(self, x):
        return aslinearoperator(super().__rmul__(x))

    def __pow__(self, p):
        return aslinearoperator(super().__pow__(p))

    def __add__(self, x):
        return aslinearoperator(super().__add__(x))

    def __neg__(self):
        return aslinearoperator(super().__neg__())

    def __sub__(self, x):
        return aslinearoperator(super().__sub__(x))

    def _adjoint(self):
        return aslinearoperator(super()._adjoint())

    def div(self, y, niter=100):
        r"""Solve the linear problem :math:`\mathbf{y}=\mathbf{A}\mathbf{x}`.

        Overloading of operator ``/`` to improve expressivity of `Pylops`
        when solving inverse problems.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data
        niter : :obj:`int`, optional
            Number of iterations (to be used only when ``explicit=False``)

        Returns
        -------
        xest : :obj:`np.ndarray`
            Estimated model

        """
        xest = self.__truediv__(y, niter=niter)
        return xest

    def __truediv__(self, y, niter=100):
        if self.explicit is True:
            if isinstance(self.A, np.ndarray):
                if self.A.shape[0] == self.A.shape[1]:
                    xest = solve(self.A, y)
                else:
                    xest = lstsq(self.A, y)[0]
            else:
                xest = spsolve(self.A, y)
        else:
            xest = lsqr(self, y, iter_lim=niter)[0]
        return xest

    def todense(self):
        r"""Return dense matrix.

        The operator in converted into its dense matrix equivalent. In order
        to do so, the operator is applied to an identity matrix whose number
        of rows and columns is equivalent to the number of columns of the
        operator. Note that this operation may be costly for very large
        operators and it is only suggest it to use as a way to inspect the
        structure of the matricial equivalent of the operator.

        Returns
        -------
        matrix : :obj:`numpy.ndarray`
            Dense matrix.

        """
        # Wrap self into a LinearOperator. This is done for cases where self
        # is a _SumLinearOperator or _ProductLinearOperator, so that it regains
        # the dense method
        Op = aslinearoperator(self)

        identity = np.eye(self.shape[1], dtype=self.dtype)
        matrix = Op.matmat(identity)
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
        i, j = indices[:,0], indices[:,1]

        # construct a sparse, CSR matrix from the entries / indices data.
        matrix = csr_matrix((entries, (j, i)), shape=self.shape, dtype=self.dtype)
        return matrix

    def eigs(self, neigs=None, symmetric=False, niter=None,
             uselobpcg=False, **kwargs_eig):
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
                        raise ValueError('cannot use scipy.sparse.linalg.lobpcg '
                                         'for non-symmetric square matrices of '
                                         'complex type...')
                    if symmetric and uselobpcg:
                        X = np.random.rand(self.shape[0], neigs)
                        eigenvalues = \
                            sp_lobpcg(self.A, X=X, maxiter=niter,
                                      **kwargs_eig)[0]
                    elif symmetric:
                        eigenvalues = sp_eigsh(self.A, k=neigs, maxiter=niter,
                                               **kwargs_eig)[0]
                    else:
                        eigenvalues = sp_eigs(self.A, k=neigs, maxiter=niter,
                                              **kwargs_eig)[0]

            else:
                if neigs is None or neigs == self.shape[1]:
                    eigenvalues = np.sqrt(eigvals(np.dot(np.conj(self.A.T),
                                                         self.A)))
                else:
                    if uselobpcg:
                        X = np.random.rand(self.shape[1], neigs)
                        eigenvalues = np.sqrt(sp_lobpcg(
                            np.dot(np.conj(self.A.T), self.A),
                            X=X, maxiter=niter, **kwargs_eig)[0])
                    else:
                        eigenvalues = np.sqrt(sp_eigsh(
                            np.dot(np.conj(self.A.T), self.A),
                            k=neigs, maxiter=niter, **kwargs_eig)[0])
        else:
            if neigs is None or neigs >= self.shape[1]:
                neigs = self.shape[1] - 2
            if self.shape[0] == self.shape[1]:
                if not symmetric and np.iscomplexobj(self) and uselobpcg:
                    raise ValueError('cannot use scipy.sparse.linalg.lobpcg for '
                                     'non symmetric square matrices of '
                                     'complex type...')
                if symmetric and uselobpcg:
                    X = np.random.rand(self.shape[0], neigs)
                    eigenvalues = \
                        sp_lobpcg(self, X=X, maxiter=niter, **kwargs_eig)[0]
                elif symmetric:
                    eigenvalues = sp_eigsh(self, k=neigs, maxiter=niter,
                                           **kwargs_eig)[0]
                else:
                    eigenvalues = sp_eigs(self, k=neigs, maxiter=niter,
                                          **kwargs_eig)[0]
            else:
                if uselobpcg:
                    X = np.random.rand(self.shape[1], neigs)
                    eigenvalues = np.sqrt(sp_lobpcg(self.H * self, X=X,
                                                    maxiter=niter,
                                                    **kwargs_eig)[0])
                else:
                    eigenvalues = np.sqrt(sp_eigs(self.H * self, k=neigs,
                                                  maxiter=niter,
                                                  **kwargs_eig)[0])
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
            cond = np.asscalar(self.eigs(neigs=1, which='LM', **kwargs_eig))/ \
                   np.asscalar(self.eigs(neigs=1, which='SM', **kwargs_eig))
        else:
            print('here')
            cond = np.asscalar(self.eigs(neigs=1, uselobpcg=True, largest=True,
                                         **kwargs_eig)) / \
                   np.asscalar(self.eigs(neigs=1, uselobpcg=True, largest=False,
                                         **kwargs_eig))

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


class _ConjLinearOperator(LinearOperator):
    """Complex conjugate linear operator
    """
    def __init__(self, Op):
        if not isinstance(Op, spLinearOperator):
            raise TypeError('Op must be a LinearOperator')
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
            raise TypeError('Op must be a LinearOperator')
        super(_ColumnLinearOperator, self).__init__(Op, Op.explicit)
        self.Op = Op
        self.cols = cols
        self.shape = (Op.shape[0], len(cols))
        if self.explicit:
            self.Opcol = Op.A[:, cols]

    def _matvec(self, x):
        if self.explicit:
            y = self.Opcol @ x
        else:
            y = np.zeros(self.Op.shape[1], dtype=self.dtype)
            y[self.cols] = x
            y = self.Op._matvec(y)
        return y

    def _rmatvec(self, x):
        if self.explicit:
            y = self.Opcol.T @ x
        else:
            y = self.Op._rmatvec(x)
            y = y[self.cols]
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