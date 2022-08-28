__all__ = ["scalability_test"]

import time
from typing import List, Tuple

import numpy.typing as npt


def scalability_test(
    Op,
    x: npt.ArrayLike,
    workers: List[int] = [1, 2, 4],
    forward: bool = True,
) -> Tuple[List[float], List[float]]:
    r"""Scalability test.

    Small auxiliary routine to test the performance of operators using
    ``multiprocessing``. This helps identifying the maximum number of workers
    beyond which no performance gain is observed.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Operator to test. It must allow for multiprocessing.
    x : :obj:`numpy.ndarray`, optional
        Input vector.
    workers : :obj:`list`, optional
        Number of workers to test out.
    forward : :obj:`bool`, optional
        Apply forward (``True``) or adjoint (``False``)

    Returns
    -------
    compute_times : :obj:`list`
        Compute times as function of workers
    speedup : :obj:`list`
        Speedup as function of workers

    """
    compute_times = []
    speedup = []
    for nworkers in workers:
        print(f"Working with {nworkers} workers...")
        # update number of workers in operator
        Op.nproc = nworkers
        # run forward/adjoint computation
        starttime = time.time()
        if forward:
            _ = Op.matvec(x)
        else:
            _ = Op.rmatvec(x)
        elapsedtime = time.time() - starttime
        compute_times.append(elapsedtime)
        speedup.append(compute_times[0] / elapsedtime)
    Op.pool.close()
    return compute_times, speedup
