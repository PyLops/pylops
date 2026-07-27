from __future__ import annotations

__all__ = ["MultiOperator"]

import concurrent.futures as mt
import multiprocessing as mp
import threading

from pylops import LinearOperator
from pylops.utils.typing import NDArray, Tparallel_kind, Tpool


class MultiOperator(LinearOperator):
    """Multiprocess/threading operator

    This class acts as a base class for all operators that wish to
    support multiprocessing/multithreading in their ``matvec``/``rmatvec``
    methods.

    Implements basic methods to instantiate and tear down a pool or workers,
    and ``_matvec``/``_rmatvec`` interfaces that dispatch to the actual
    implementations for serial/multiprocess/multithread, namely:

    - ``_matvec_serial`` / ``_rmatvec_serial``: serial implementation
    - ``_matvec_multiproc`` / ``_rmatvec_multiproc``: multiprocess implementation
    - ``_matvec_multithread`` / ``_matvec_multithread``: multithreading implementation

    Developers are in charge of implementing these methods for specific operators or
    overwriting ``_matvec``/``_rmatvec`` if not all of the implementations are available.

    .. note:: End users of PyLops should not use this class directly but simply
          use operators that are already implemented. This class is meant for
          developers and it has to be used as the parent class of any new operator
          with multiprocess/multithreading capabilities developed within PyLops.

    """

    def _setup_pool(
        self,
        nproc: int = 1,
        parallel_kind: Tparallel_kind = "multiproc",
    ) -> None:
        """Setup pool for multiprocessing/multithreading"""
        if parallel_kind not in ["multiproc", "multithread"]:
            msg = "parallel_kind must be 'multiproc' or 'multithread'"
            raise ValueError(msg)

        # create pool for multithreading / multiprocessing
        self.parallel_kind = parallel_kind
        self._nproc = nproc
        self.pool: Tpool | None = None
        if self.nproc > 1:
            if self.parallel_kind == "multiproc":
                self.pool = mp.Pool(processes=nproc)
            else:
                self.pool = mt.ThreadPoolExecutor(max_workers=nproc)
                self.lock = threading.Lock()

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

    def close(self) -> None:
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
