__all__ = ["HStack"]

import multiprocessing as mp

import numpy as np
import scipy as sp

# need to check scipy version since the interface submodule changed into
# _interface from scipy>=1.8.0
sp_version = sp.__version__.split(".")
if int(sp_version[0]) <= 1 and int(sp_version[1]) < 8:
    from scipy.sparse.linalg.interface import LinearOperator as spLinearOperator
    from scipy.sparse.linalg.interface import _get_dtype
else:
    from scipy.sparse.linalg._interface import _get_dtype
    from scipy.sparse.linalg._interface import (
        LinearOperator as spLinearOperator,
    )

from typing import Optional, Sequence

from pylops import LinearOperator
from pylops.basicoperators import MatrixMult
from pylops.utils.backend import get_array_module, inplace_add, inplace_set
from pylops.utils.typing import NDArray


def _matvec_rmatvec_map(op, x: NDArray) -> NDArray:
    """matvec/rmatvec for multiprocessing"""
    return op(x).squeeze()


class HStack(LinearOperator):
    r"""Horizontal stacking.

    Stack a set of N linear operators horizontally.

    Parameters
    ----------
    ops : :obj:`list`
        Linear operators to be stacked. Alternatively,
        :obj:`numpy.ndarray` or :obj:`scipy.sparse` matrices can be passed
        in place of one or more operators.
    nproc : :obj:`int`, optional
        Number of processes used to evaluate the N operators in parallel
        using ``multiprocessing``. If ``nproc=1``, work in serial mode.
    forceflat : :obj:`bool`, optional
        .. versionadded:: 2.2.0

        Force an array to be flattened after matvec.
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
        If ``ops`` have different number of columns

    Notes
    -----
    An horizontal stack of N linear operators is created such as its
    application in forward mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L}_{1} & \mathbf{L}_{2} & \ldots & \mathbf{L}_{N}
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{x}_{1}  \\
            \mathbf{x}_{2}  \\
            \vdots     \\
            \mathbf{x}_{N}
        \end{bmatrix} =
        \mathbf{L}_{1} \mathbf{x}_1 + \mathbf{L}_{2} \mathbf{x}_2 +
        \ldots + \mathbf{L}_{N} \mathbf{x}_N

    while its application in adjoint mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L}_{1}^H  \\
            \mathbf{L}_{2}^H  \\
            \vdots     \\
            \mathbf{L}_{N}^H
        \end{bmatrix}
        \mathbf{y} =
        \begin{bmatrix}
            \mathbf{L}_{1}^H \mathbf{y}  \\
            \mathbf{L}_{2}^H \mathbf{y}  \\
            \vdots     \\
            \mathbf{L}_{N}^H \mathbf{y}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{x}_{1}  \\
            \mathbf{x}_{2}  \\
            \vdots     \\
            \mathbf{x}_{N}
        \end{bmatrix}

    """

    def __init__(
        self,
        ops: Sequence[LinearOperator],
        nproc: int = 1,
        forceflat: bool = None,
        dtype: Optional[str] = None,
    ) -> None:
        self.ops = ops
        mops = np.zeros(len(ops), dtype=int)
        for iop, oper in enumerate(ops):
            if not isinstance(oper, (LinearOperator, spLinearOperator)):
                self.ops[iop] = MatrixMult(oper, dtype=oper.dtype)
            mops[iop] = self.ops[iop].shape[1]
        self.mops = int(mops.sum())
        nops = [oper.shape[0] for oper in self.ops]
        if len(set(nops)) > 1:
            raise ValueError("operators have different number of rows")
        self.nops = int(nops[0])
        self.mmops = np.insert(np.cumsum(mops), 0, 0)
        # define dimsd (check if all operators have the same,
        # otherwise make same as self.nops and forceflat=True)
        dimsd = [op.dimsd for op in self.ops]
        if len(set(dimsd)) == 1:
            dimsd = dimsd[0]
        else:
            dimsd = (self.nops,)
            forceflat = True
        # create pool for multiprocessing
        self._nproc = nproc
        self.pool = None
        if self.nproc > 1:
            self.pool = mp.Pool(processes=nproc)
        dtype = _get_dtype(self.ops) if dtype is None else np.dtype(dtype)
        clinear = all([getattr(oper, "clinear", True) for oper in self.ops])
        super().__init__(
            dtype=dtype,
            shape=(self.nops, self.mops),
            dimsd=dimsd,
            clinear=clinear,
            forceflat=forceflat,
        )

    @property
    def nproc(self) -> int:
        return self._nproc

    @nproc.setter
    def nproc(self, nprocnew: int):
        if self._nproc > 1:
            self.pool.close()
        if nprocnew > 1:
            self.pool = mp.Pool(processes=nprocnew)
        self._nproc = nprocnew

    def _matvec_serial(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(self.nops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y = inplace_add(
                oper.matvec(x[self.mmops[iop] : self.mmops[iop + 1]]).squeeze(),
                y,
                slice(None, None),
            )
        return y

    def _rmatvec_serial(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros(self.mops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y = inplace_set(
                oper.rmatvec(x).squeeze(),
                y,
                slice(self.mmops[iop], self.mmops[iop + 1]),
            )
        return y

    def _matvec_multiproc(self, x: NDArray) -> NDArray:
        ys = self.pool.starmap(
            _matvec_rmatvec_map,
            [
                (oper._matvec, x[self.mmops[iop] : self.mmops[iop + 1]])
                for iop, oper in enumerate(self.ops)
            ],
        )
        y = np.sum(ys, axis=0)
        return y

    def _rmatvec_multiproc(self, x: NDArray) -> NDArray:
        ys = self.pool.starmap(
            _matvec_rmatvec_map,
            [(oper._rmatvec, x) for iop, oper in enumerate(self.ops)],
        )
        y = np.hstack(ys)
        return y

    def _matvec(self, x: NDArray) -> NDArray:
        if self.nproc == 1:
            y = self._matvec_serial(x)
        else:
            y = self._matvec_multiproc(x)
        return y

    def _rmatvec(self, x: NDArray) -> NDArray:
        if self.nproc == 1:
            y = self._rmatvec_serial(x)
        else:
            y = self._rmatvec_multiproc(x)
        return y
