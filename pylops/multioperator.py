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

    Implements basic methods to instantiate and tear down a pool of workers
    and ``_matvec``/``_rmatvec`` interfaces that dispatch to the actual
    implementations for serial/multiprocess/multithread, namely:

    - ``_matvec_serial`` / ``_rmatvec_serial``: serial implementation;
    - ``_matvec_multiproc`` / ``_rmatvec_multiproc``: multiprocess implementation;
    - ``_matvec_multithread`` / ``_matvec_multithread``: multithreading implementation.

    Developers are in charge of implementing these methods for specific operators or
    overwriting ``_matvec``/``_rmatvec`` if not all of the implementations are
    available. Moreover, derived classes must call ``_setup_pool`` in their constructor
    (before ``super().__init__``), as this method defines the ``nproc``, ``parallel_kind``
    and ``pool`` attributes used by all other methods.

    Note that ``nproc`` can be changed after construction: this will stop the current
    worker pool and restart a new one. The chosen ``parallel_kind`` is retained
    when changing ``nproc``. On the other hand, ``parallel_kind`` cannot be changed
    after construction.

    .. note:: End users of PyLops should not use this class directly but simply
          use operators that are already implemented. This class is meant for
          developers and it has to be used as the parent class of any new operator
          with multiprocess/multithreading capabilities developed within PyLops.

    """

    def _start_pool(self, nproc: int) -> None:
        """Start pool of ``nproc`` workers for multiprocessing/multithreading"""
        if self.parallel_kind == "multiproc":
            self.pool = mp.Pool(processes=nproc)
        else:
            self.pool = mt.ThreadPoolExecutor(max_workers=nproc)
            self.lock = threading.Lock()

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
        self._parallel_kind = parallel_kind
        self._nproc = nproc
        self.pool: Tpool | None = None
        if self.nproc > 1:
            self._start_pool(nproc)

    @property
    def nproc(self) -> int:
        """Number of workers; assigning it rebuilds or closes the worker pool."""
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
            self._start_pool(nprocnew)
        self._nproc = nprocnew

    @property
    def parallel_kind(self) -> Tparallel_kind:
        """Kind of parallelism (``multiproc`` or ``multithread``), fixed at construction."""
        return self._parallel_kind

    @parallel_kind.setter
    def parallel_kind(self, _: Tparallel_kind) -> None:
        msg = "parallel_kind cannot be changed after construction"
        raise AttributeError(msg)

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
