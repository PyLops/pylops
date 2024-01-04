__all__ = [
    "sliding2d_design",
    "Sliding2D",
]

import logging
from typing import Tuple

import numpy as np

from pylops import LinearOperator
from pylops.basicoperators import BlockDiag, Diagonal, HStack, Restriction
from pylops.utils.tapers import taper2d
from pylops.utils.typing import InputDimsLike, NDArray

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def _slidingsteps(
    ntr: int,
    nwin: int,
    nover: int,
) -> Tuple[NDArray, NDArray]:
    """Identify sliding window initial and end points given overall
    trace length, window length and overlap

    Parameters
    ----------
    ntr : :obj:`int`
        Number of samples in trace
    nwin : :obj:`int`
        Number of samples of window
    nover : :obj:`int`
        Number of samples of overlapping part of window

    Returns
    -------
    starts : :obj:`np.ndarray`
        Start indices
    ends : :obj:`np.ndarray`
        End indices

    """
    if nwin > ntr:
        raise ValueError(f"nwin={nwin} is bigger than ntr={ntr}...")
    step = nwin - nover
    starts = np.arange(0, ntr - nwin + 1, step, dtype=int)
    ends = starts + nwin
    return starts, ends


def sliding2d_design(
    dimsd: Tuple[int, int],
    nwin: int,
    nover: int,
    nop: Tuple[int, int],
) -> Tuple[int, Tuple[int, int], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
    """Design Sliding2D operator

    This routine can be used prior to creating the :class:`pylops.signalprocessing.Sliding2D`
    operator to identify the correct number of windows to be used based on the dimension of the data (``dimsd``),
    dimension of the window (``nwin``), overlap (``nover``),a and dimension of the operator acting in the model
    space.

    Parameters
    ----------
    dimsd : :obj:`tuple`
        Shape of 2-dimensional data.
    nwin : :obj:`int`
        Number of samples of window.
    nover : :obj:`int`
        Number of samples of overlapping part of window.
    nop : :obj:`tuple`
        Size of model in the transformed domain.

    Returns
    -------
    nwins : :obj:`int`
        Number of windows.
    dims : :obj:`tuple`
        Size of 2-dimensional model.
    mwins_inends : :obj:`tuple`
        Start and end indices for model patches (stored as tuple of tuples).
    dwins_inends : :obj:`tuple`
        Start and end indices for data patches (stored as tuple of tuples).

    """
    # data windows
    dwin_ins, dwin_ends = _slidingsteps(dimsd[0], nwin, nover)
    dwins_inends = (dwin_ins, dwin_ends)
    nwins = len(dwin_ins)

    # model windows
    dims = (nwins * nop[0], nop[1])
    mwin_ins, mwin_ends = _slidingsteps(dims[0], nop[0], 0)
    mwins_inends = (mwin_ins, mwin_ends)

    # print information about patching
    logging.warning("%d windows required...", nwins)
    logging.warning(
        "data wins - start:%s, end:%s",
        dwin_ins,
        dwin_ends,
    )
    logging.warning(
        "model wins - start:%s, end:%s",
        mwin_ins,
        mwin_ends,
    )
    return nwins, dims, mwins_inends, dwins_inends


def Sliding2D(
    Op: LinearOperator,
    dims: InputDimsLike,
    dimsd: InputDimsLike,
    nwin: int,
    nover: int,
    tapertype: str = "hanning",
    name: str = "S",
) -> LinearOperator:
    """2D Sliding transform operator.

    Apply a transform operator ``Op`` repeatedly to slices of the model
    vector in forward mode and slices of the data vector in adjoint mode.
    More specifically, in forward mode the model vector is divided into
    slices, each slice is transformed, and slices are then recombined in a
    sliding window fashion. Both model and data are internally reshaped and
    interpreted as 2-dimensional arrays: each slice contains a portion
    of the array in the first dimension (and the entire second dimension).

    This operator can be used to perform local, overlapping transforms (e.g.,
    :obj:`pylops.signalprocessing.FFT2D`
    or :obj:`pylops.signalprocessing.Radon2D`) on 2-dimensional arrays.

    .. note:: The shape of the model has to be consistent with
       the number of windows for this operator not to return an error. As the
       number of windows depends directly on the choice of ``nwin`` and
       ``nover``, it is recommended to first run ``sliding2d_design`` to obtain
       the corresponding ``dims`` and number of windows.

    .. warning:: Depending on the choice of `nwin` and `nover` as well as the
       size of the data, sliding windows may not cover the entire data.
       The start and end indices of each window will be displayed and returned
       with running ``sliding2d_design``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Transform operator
    dims : :obj:`tuple`
        Shape of 2-dimensional model. Note that ``dims[0]`` should be multiple
        of the model size of the transform in the first dimension
    dimsd : :obj:`tuple`
        Shape of 2-dimensional data
    nwin : :obj:`int`
        Number of samples of window
    nover : :obj:`int`
        Number of samples of overlapping part of window
    tapertype : :obj:`str`, optional
        Type of taper (``hanning``, ``cosine``, ``cosinesquare`` or ``None``)
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Returns
    -------
    Sop : :obj:`pylops.LinearOperator`
        Sliding operator

    Raises
    ------
    ValueError
        Identified number of windows is not consistent with provided model
        shape (``dims``).

    """
    # data windows
    dwin_ins, dwin_ends = _slidingsteps(dimsd[0], nwin, nover)
    nwins = len(dwin_ins)

    # check patching
    if nwins * Op.shape[1] // dims[1] != dims[0]:
        raise ValueError(
            f"Model shape (dims={dims}) is not consistent with chosen "
            f"number of windows. Run sliding2d_design to identify the "
            f"correct number of windows for the current "
            "model size..."
        )

    # create tapers
    if tapertype is not None:
        tap = taper2d(dimsd[1], nwin, nover, tapertype=tapertype).astype(Op.dtype)
        tapin = tap.copy()
        tapin[:nover] = 1
        tapend = tap.copy()
        tapend[-nover:] = 1
        taps = {}
        taps[0] = tapin if nwins > 1 else tap
        for i in range(1, nwins - 1):
            taps[i] = tap
        taps[nwins - 1] = tapend if nwins > 1 else tap

    # transform to apply
    if tapertype is None:
        OOp = BlockDiag([Op for _ in range(nwins)])
    else:
        OOp = BlockDiag(
            [Diagonal(taps[itap].ravel(), dtype=Op.dtype) * Op for itap in range(nwins)]
        )

    combining = HStack(
        [
            Restriction(dimsd, range(win_in, win_end), axis=0, dtype=Op.dtype).H
            for win_in, win_end in zip(dwin_ins, dwin_ends)
        ]
    )
    Sop = LinearOperator(combining * OOp)
    Sop.dims, Sop.dimsd = (nwins, int(dims[0] // nwins), dims[1]), dimsd
    Sop.name = name
    return Sop
