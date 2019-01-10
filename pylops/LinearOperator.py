from __future__ import division

import numpy as np
from scipy.linalg import solve, lstsq
from scipy.sparse.linalg import LinearOperator as spLinearOperator
from scipy.sparse.linalg import spsolve, lsqr
from scipy.linalg import eigvals
from scipy.sparse.linalg import eigs as sp_eigs
from scipy.sparse.linalg import eigsh as sp_eigsh

class LinearOperator(spLinearOperator):
    """Common interface for performing matrix-vector products.

    This class is an overload of the
    :py:class:`scipy.sparse.linalg.LinearOperator` class, which contains
    additional overloading to standard operators such as ``__div__``.

    """
    def __init__(self, explicit=False):
        super(LinearOperator, self).__init__()
        self.explicit = explicit

    def div(self, y):
        r"""Solve the linear problem :math:`\mathbf{y}=\mathbf{A}\mathbf{x}`.

        Overloading of operator ``/`` to improve expressivity of `Pylops`
        when solving inverse problems.

        Parameters
        ----------
        y : :obj:`np.ndarray`
            Data

        Returns
        -------
        xest : :obj:`np.ndarray`
            Estimated model

        """
        xest = self.__truediv__(y)
        return xest

    def __truediv__(self, y):
        if self.explicit is True:
            if isinstance(self.A, np.ndarray):
                if self.A.shape[0] == self.A.shape[1]:
                    xest = solve(self.A, y)
                else:
                    xest = lstsq(self.A, y)[0]
            else:
                xest = spsolve(self.A, y)
        else:
            xest = lsqr(self, y)[0]
        return xest

    def eigs(self, neigs=None, **kwargs_eig):
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
        **kwargs_eig
            Arbitrary keyword arguments for
            :func:`scipy.sparse.linalg.eigs` or
            :func:`scipy.sparse.linalg.eigsh`

        Returns
        -------
        eigenvalues : :obj:`numpy.ndarray`
            Operator eigenvalues.

        Notes
        -----
        Eigenvalues are estimated using :func:`scipy.sparse.linalg.eigs`
        (``explicit=True``) or :func:`scipy.sparse.linalg.eigsh`
        (``explicit=False``).

        This is a port of ARPACK [1]_, a Fortran package which provides
        routines for quickly finding eigenvalues/eigenvectors
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
                    eigenvalues = sp_eigs(self.A, k=neigs, **kwargs_eig)[0]
            else:
                if neigs is None or neigs == self.shape[1]:
                    eigenvalues = np.sqrt(eigvals(np.dot(np.conj(self.A.T),
                                                         self.A)))
                else:
                    eigenvalues = np.sqrt(sp_eigsh(
                        np.dot(np.conj(self.A.T), self.A),
                        k=neigs, **kwargs_eig)[0])
        else:
            if neigs is None or neigs >= self.shape[1]:
                neigs = self.shape[1]-2
            if self.shape[0] == self.shape[1]:
                eigenvalues = sp_eigs(self, k=neigs, **kwargs_eig)[0]
            else:
                eigenvalues = np.sqrt(sp_eigs(self.H * self,
                                              k=neigs, **kwargs_eig)[0])
        return -np.sort(-eigenvalues)

    def cond(self, **kwargs_eig):
        r"""Condition number of linear operator.

        Return an estimate of the condition number of the linear operator as
        the ratio of the largest and lowest estimated eigenvalues.

        Parameters
        ----------
        **kwargs_eig
            Arbitrary keyword arguments for
            :func:`scipy.sparse.linalg.eigs` or
            :func:`scipy.sparse.linalg.eigsh`

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
        cond = np.asscalar(self.eigs(neigs=1, which='LM', **kwargs_eig))/ \
               np.asscalar(self.eigs(neigs=1, which='SM', **kwargs_eig))
        return cond
