__all__ = ["BlockDiag"]

import concurrent.futures as mt
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
        _get_dtype,
        LinearOperator as spLinearOperator,
    )

from typing import Optional, Sequence

from pylops import LinearOperator
from pylops.basicoperators import MatrixMult
from pylops.utils.backend import get_array_module, get_module, inplace_set
from pylops.utils.typing import DTypeLike, NDArray, Tinoutengine, Tparallel_kind


def _matvec_rmatvec_map(op, x: NDArray) -> NDArray:
    """matvec/rmatvec for multiprocessing"""
    return op(x).squeeze()


class BlockDiag(LinearOperator):
    r"""Block-diagonal operator.

    Create a block-diagonal operator from N linear operators.

    Parameters
    ----------
    ops : :obj:`list`
        Linear operators to be stacked. Alternatively,
        :obj:`numpy.ndarray` or :obj:`scipy.sparse` matrices can be passed
        in place of one or more operators.
    nproc : :obj:`int`, optional
        Number of processes/threads used to evaluate the N operators in parallel using
        ``multiprocessing``/``concurrent.futures``. If ``nproc=1``, work in serial mode.
    forceflat : :obj:`bool`, optional
        .. versionadded:: 2.2.0

        Force an array to be flattened after matvec and rmatvec.
    inoutengine : :obj:`tuple`, optional
        .. versionadded:: 2.4.0

        Type of output vectors of `matvec` and `rmatvec. If ``None``, this is
        inferred directly from the input vectors. Note that this is ignored
        if ``nproc>1``.
    parallel_kind : :obj:`str`, optional
        .. versionadded:: 2.6.0

        Parallelism kind when ``nproc>1``. Can be ``multiproc`` (using
        :mod:`multiprocessing`) or ``multithread`` (using
        :class:`concurrent.futures.ThreadPoolExecutor`). Defaults
        to ``multiproc``.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    nops : :obj:`int`
        Number of rows of the full operator (sum of rows of each block).
    mops : :obj:`int`
        Number of columns of the full operator (sum of columns of each block).
    nnops : :obj:`numpy.ndarray`
        Cumulative sum of rows of each block, with a leading zero.
    mmops : :obj:`numpy.ndarray`
        Cumulative sum of columns of each block, with a leading zero.
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening.

        For example, ``y_reshaped = (Op * x.ravel()).reshape(Op.dimsd)``.
    pool : :obj:`multiprocessing.Pool` or :obj:`concurrent.futures.ThreadPoolExecutor` or :obj:`None`
        Pool of workers used to evaluate the N operators in parallel.
        When ``nproc=1``, no pool is created (i.e., ``pool=None``).
    shape : :obj:`tuple`
        Operator shape.

    Notes
    -----
    A block-diagonal operator composed of N linear operators is created such
    as its application in forward mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L}_1  & \mathbf{0}   &  \ldots &  \mathbf{0}  \\
            \mathbf{0}    & \mathbf{L}_2 &  \ldots &  \mathbf{0}  \\
            \vdots        & \vdots       &  \ddots &  \vdots         \\
            \mathbf{0}    & \mathbf{0}   &  \ldots &  \mathbf{L}_N
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{x}_{1}  \\
            \mathbf{x}_{2}  \\
            \vdots     \\
            \mathbf{x}_{N}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{L}_1 \mathbf{x}_{1}  \\
            \mathbf{L}_2 \mathbf{x}_{2}  \\
            \vdots     \\
            \mathbf{L}_N \mathbf{x}_{N}
        \end{bmatrix}

    while its application in adjoint mode leads to

    .. math::
        \begin{bmatrix}
            \mathbf{L}_1^H  & \mathbf{0}     & \ldots & \mathbf{0}  \\
            \mathbf{0}      & \mathbf{L}_2^H & \ldots & \mathbf{0}  \\
            \vdots          & \vdots         & \ddots & \vdots      \\
            \mathbf{0}      & \mathbf{0}     & \ldots & \mathbf{L}_N^H
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{y}_{1}  \\
            \mathbf{y}_{2}  \\
            \vdots     \\
            \mathbf{y}_{N}
        \end{bmatrix} =
        \begin{bmatrix}
            \mathbf{L}_1^H \mathbf{y}_{1}  \\
            \mathbf{L}_2^H \mathbf{y}_{2}  \\
            \vdots     \\
            \mathbf{L}_N^H \mathbf{y}_{N}
        \end{bmatrix}

    """

    def __init__(
        self,
        ops: Sequence[LinearOperator],
        nproc: int = 1,
        forceflat: Optional[bool] = None,
        inoutengine: Optional[Tinoutengine] = None,
        parallel_kind: Tparallel_kind = "multiproc",
        dtype: Optional[DTypeLike] = None,
    ) -> None:
        if parallel_kind not in ["multiproc", "multithread"]:
            raise ValueError("parallel_kind must be 'multiproc' or 'multithread'")
        # identify dimensions
        self.ops = ops
        mops = np.zeros(len(ops), dtype=int)
        nops = np.zeros(len(ops), dtype=int)
        for iop, oper in enumerate(ops):
            if not isinstance(oper, (LinearOperator, spLinearOperator)):
                self.ops[iop] = MatrixMult(oper, dtype=oper.dtype)
            nops[iop] = self.ops[iop].shape[0]
            mops[iop] = self.ops[iop].shape[1]
        self.nops = int(nops.sum())
        self.mops = int(mops.sum())
        self.nnops = np.insert(np.cumsum(nops), 0, 0)
        self.mmops = np.insert(np.cumsum(mops), 0, 0)
        # define dims (check if all operators have the same,
        # otherwise make same as self.mops and forceflat=True)
        dims = [op.dims for op in self.ops]
        if len(set(dims)) == 1:
            dims = (len(ops), *dims[0])
        else:
            dims = (self.mops,)
            forceflat = True
        # define dimsd (check if all operators have the same,
        # otherwise make same as self.nops and forceflat=True)
        dimsd = [op.dimsd for op in self.ops]
        if len(set(dimsd)) == 1:
            dimsd = (len(ops), *dimsd[0])
        else:
            dimsd = (self.nops,)
            forceflat = True
        # create pool for multithreading / multiprocessing
        self.parallel_kind = parallel_kind
        self._nproc = nproc
        self.pool: Optional[mp.pool.Pool] = None
        if self.nproc > 1:
            if self.parallel_kind == "multiproc":
                self.pool = mp.Pool(processes=nproc)
            else:
                self.pool = mt.ThreadPoolExecutor(max_workers=nproc)
        self.inoutengine = inoutengine
        dtype = _get_dtype(ops) if dtype is None else np.dtype(dtype)
        clinear = all([getattr(oper, "clinear", True) for oper in self.ops])
        super().__init__(
            dtype=dtype,
            dims=dims,
            dimsd=dimsd,
            clinear=clinear,
            forceflat=forceflat,
        )

    @property
    def nproc(self) -> int:
        return self._nproc

    @nproc.setter
    def nproc(self, nprocnew: int) -> None:
        if self._nproc > 1 and self.pool is not None:
            if self.parallel_kind == "multiproc":
                self.pool.close()
                self.pool.join()
            else:
                self.pool.shutdown()
        if nprocnew > 1:
            if self.parallel_kind == "multiproc":
                self.pool = mp.Pool(processes=nprocnew)
            else:
                self.pool = mt.ThreadPoolExecutor(max_workers=nprocnew)
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
                oper.matvec(x[self.mmops[iop] : self.mmops[iop + 1]]).squeeze(),
                y,
                slice(self.nnops[iop], self.nnops[iop + 1]),
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
            y = inplace_set(
                oper.rmatvec(x[self.nnops[iop] : self.nnops[iop + 1]]).squeeze(),
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
        y = np.hstack(ys)
        return y

    def _matvec_multithread(self, x: NDArray) -> NDArray:
        ys = list(
            self.pool.map(
                lambda args: _matvec_rmatvec_map(*args),
                [
                    (oper._matvec, x[self.mmops[iop] : self.mmops[iop + 1]])
                    for iop, oper in enumerate(self.ops)
                ],
            )
        )
        y = np.hstack(ys)
        return y

    def _rmatvec_multithread(self, x: NDArray) -> NDArray:
        ys = list(
            self.pool.map(
                lambda args: _matvec_rmatvec_map(*args),
                [
                    (oper._rmatvec, x[self.nnops[iop] : self.nnops[iop + 1]])
                    for iop, oper in enumerate(self.ops)
                ],
            )
        )
        y = np.hstack(ys)
        return y

    def _matvec(self, x: NDArray) -> NDArray:
        if self.nproc == 1:
            y = self._matvec_serial(x)
        else:
            if self.parallel_kind == "multiproc":
                y = self._matvec_multiproc(x)
            else:
                y = self._matvec_multithread(x)
        return y

    def _rmatvec(self, x: NDArray) -> NDArray:
        if self.nproc == 1:
            y = self._rmatvec_serial(x)
        else:
            if self.parallel_kind == "multiproc":
                y = self._rmatvec_multiproc(x)
            else:
                y = self._rmatvec_multithread(x)
        return y

    def close(self):
        """Close the pool of workers used for multiprocessing
        / multithreading.
        """
        if self.pool is not None:
            if self.parallel_kind == "multiproc":
                self.pool.close()
                self.pool.join()
            else:
                self.pool.shutdown()
            self.pool = None
