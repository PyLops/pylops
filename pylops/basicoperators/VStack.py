import multiprocessing as mp

import numpy as np
import scipy as sp
from scipy.sparse.linalg.interface import LinearOperator as spLinearOperator

# need to check scipy version since the interface submodule changed into
# _interface from scipy>=1.8.0
sp_version = sp.__version__.split(".")
if int(sp_version[0]) <= 1 and int(sp_version[1]) < 8:
    from scipy.sparse.linalg.interface import _get_dtype
else:
    from scipy.sparse.linalg._interface import _get_dtype

from pylops import LinearOperator
from pylops.basicoperators import MatrixMult
from pylops.utils.backend import get_array_module


def _matvec_rmatvec_map(op, x):
    """matvec/rmatvec for multiprocessing"""
    return op(x).squeeze()


class VStack(LinearOperator):
    r"""Vertical stacking.

    Stack a set of N linear operators vertically.

    Parameters
    ----------
    ops : :obj:`list`
        Linear operators to be stacked. Alternatively,
        :obj:`numpy.ndarray` or :obj:`scipy.sparse` matrices can be passed
        in place of one or more operators.
    nproc : :obj:`int`, optional
        Number of processes used to evaluate the N operators in parallel using
        ``multiprocessing``. If ``nproc=1``, work in serial mode.
    dtype : :obj:`str`, optional
        Type of elements in input array.

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
        If ``ops`` have different number of rows

    Notes
    -----
    A vertical stack of N linear operators is created such as its application
    in forward mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L}_{1}  \\
            \mathbf{L}_{2}  \\
            \vdots     \\
            \mathbf{L}_{N}
        \end{bmatrix}
        \mathbf{x} =
        \begin{bmatrix}
            \mathbf{L}_{1} \mathbf{x}  \\
            \mathbf{L}_{2} \mathbf{x}  \\
            \vdots     \\
            \mathbf{L}_{N} \mathbf{x}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{y}_{1}  \\
            \mathbf{y}_{2}  \\
            \vdots     \\
            \mathbf{y}_{N}
        \end{bmatrix}

    while its application in adjoint mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L}_{1}^H & \mathbf{L}_{2}^H & \ldots & \mathbf{L}_{N}^H
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{y}_{1}  \\
            \mathbf{y}_{2}  \\
            \vdots     \\
            \mathbf{y}_{N}
        \end{bmatrix} =
        \mathbf{L}_{1}^H \mathbf{y}_1 + \mathbf{L}_{2}^H \mathbf{y}_2 +
        \ldots + \mathbf{L}_{N}^H \mathbf{y}_N

    """

    def __init__(self, ops, nproc=1, dtype=None):
        self.ops = ops
        nops = np.zeros(len(self.ops), dtype=int)
        for iop, oper in enumerate(ops):
            if not isinstance(oper, (LinearOperator, spLinearOperator)):
                self.ops[iop] = MatrixMult(oper, dtype=oper.dtype)
            nops[iop] = self.ops[iop].shape[0]
        self.nops = int(nops.sum())
        mops = [oper.shape[1] for oper in self.ops]
        if len(set(mops)) > 1:
            raise ValueError("operators have different number of columns")
        self.mops = int(mops[0])
        self.nnops = np.insert(np.cumsum(nops), 0, 0)
        # create pool for multiprocessing
        self._nproc = nproc
        self.pool = None
        if self.nproc > 1:
            self.pool = mp.Pool(processes=nproc)
        self.shape = (self.nops, self.mops)
        if dtype is None:
            self.dtype = _get_dtype(self.ops)
        else:
            self.dtype = np.dtype(dtype)
        self.explicit = False
        self.clinear = all([getattr(oper, "clinear", True) for oper in self.ops])

    @property
    def nproc(self):
        return self._nproc

    @nproc.setter
    def nproc(self, nprocnew):
        if self._nproc > 1:
            self.pool.close()
        if nprocnew > 1:
            self.pool = mp.Pool(processes=nprocnew)
        self._nproc = nprocnew

    def _matvec_serial(self, x):
        ncp = get_array_module(x)
        y = ncp.zeros(self.nops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y[self.nnops[iop] : self.nnops[iop + 1]] = oper.matvec(x).squeeze()
        return y

    def _rmatvec_serial(self, x):
        ncp = get_array_module(x)
        y = ncp.zeros(self.mops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y += oper.rmatvec(x[self.nnops[iop] : self.nnops[iop + 1]]).squeeze()
        return y

    def _matvec_multiproc(self, x):
        ys = self.pool.starmap(
            _matvec_rmatvec_map,
            [(oper._matvec, x) for iop, oper in enumerate(self.ops)],
        )
        y = np.hstack(ys)
        return y

    def _rmatvec_multiproc(self, x):
        ys = self.pool.starmap(
            _matvec_rmatvec_map,
            [
                (oper._rmatvec, x[self.nnops[iop] : self.nnops[iop + 1]])
                for iop, oper in enumerate(self.ops)
            ],
        )
        y = np.sum(ys, axis=0)
        return y

    def _matvec(self, x):
        if self.nproc == 1:
            y = self._matvec_serial(x)
        else:
            y = self._matvec_multiproc(x)
        return y

    def _rmatvec(self, x):
        if self.nproc == 1:
            y = self._rmatvec_serial(x)
        else:
            y = self._rmatvec_multiproc(x)
        return y
