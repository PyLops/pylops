__all__ = ["VStack"]

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
    from scipy.sparse.linalg._interface import (
        LinearOperator as spLinearOperator,
    )
    from scipy.sparse.linalg._interface import _get_dtype

from typing import Callable, Optional, Sequence

from pylops import LinearOperator
from pylops.basicoperators import MatrixMult
from pylops.utils.backend import get_array_module, get_module, inplace_add, inplace_set
from pylops.utils.typing import DTypeLike, NDArray


def _matvec_rmatvec_map(op: Callable, x: NDArray) -> NDArray:
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
    forceflat : :obj:`bool`, optional
        .. versionadded:: 2.2.0

        Force an array to be flattened after rmatvec.
    inoutengine : :obj:`tuple`, optional
        .. versionadded:: 2.4.0

        Type of output vectors of `matvec` and `rmatvec. If ``None``, this is
        inferred directly from the input vectors. Note that this is ignored
        if ``nproc>1``.
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

    def __init__(
        self,
        ops: Sequence[LinearOperator],
        nproc: int = 1,
        forceflat: bool = None,
        inoutengine: Optional[tuple] = None,
        dtype: Optional[DTypeLike] = None,
    ) -> None:
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
        # define dims (check if all operators have the same,
        # otherwise make same as self.mops and forceflat=True)
        dims = [op.dims for op in self.ops]
        if len(set(dims)) == 1:
            dims = dims[0]
        else:
            dims = (self.mops,)
            forceflat = True
        # create pool for multiprocessing
        self._nproc = nproc
        self.pool = None
        if self.nproc > 1:
            self.pool = mp.Pool(processes=nproc)

        self.inoutengine = inoutengine
        dtype = _get_dtype(self.ops) if dtype is None else np.dtype(dtype)
        clinear = all([getattr(oper, "clinear", True) for oper in self.ops])
        super().__init__(
            dtype=dtype,
            shape=(self.nops, self.mops),
            dims=dims,
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
        ncp = (
            get_array_module(x)
            if self.inoutengine is None
            else get_module(self.inoutengine[0])
        )
        y = ncp.zeros(self.nops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y = inplace_set(
                oper.matvec(x).squeeze(), y, slice(self.nnops[iop], self.nnops[iop + 1])
            )
        return y

    def _rmatvec_serial(self, x: NDArray) -> NDArray:
        ncp = (
            get_array_module(x)
            if self.inoutengine is None
            else get_module(self.inoutengine[1])
        )
        y = ncp.zeros(self.mops, dtype=self.dtype)
        for iop, oper in enumerate(self.ops):
            y = inplace_add(
                oper.rmatvec(x[self.nnops[iop] : self.nnops[iop + 1]]).squeeze(),
                y,
                slice(None, None),
            )
        return y

    def _matvec_multiproc(self, x: NDArray) -> NDArray:
        ys = self.pool.starmap(
            _matvec_rmatvec_map,
            [(oper._matvec, x) for iop, oper in enumerate(self.ops)],
        )
        y = np.hstack(ys)
        return y

    def _rmatvec_multiproc(self, x: NDArray) -> NDArray:
        ys = self.pool.starmap(
            _matvec_rmatvec_map,
            [
                (oper._rmatvec, x[self.nnops[iop] : self.nnops[iop + 1]])
                for iop, oper in enumerate(self.ops)
            ],
        )
        y = np.sum(ys, axis=0)
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
