import threading
from collections.abc import Callable

from pylops.utils.typing import NDArray


def _matvec_rmatvec_map(op: Callable[[NDArray], NDArray], x: NDArray) -> NDArray:
    """matvec/rmatvec for multiprocessing / multithreading"""
    return op(x).squeeze()


def _matvec_rmatvec_map_mt(
    op: Callable[[NDArray], NDArray], x: NDArray, y: NDArray, lock: threading.Lock
) -> None:
    """rmatvec for multithreading with lock"""
    ylocal = op(x).squeeze()
    with lock:
        y[:] += ylocal
