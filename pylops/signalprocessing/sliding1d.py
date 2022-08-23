__all__ = [
    "sliding1d_design",
    "Sliding1D",
]

import logging
from typing import Tuple, Union

import numpy as np

from pylops import aslinearoperator
from pylops.basicoperators import BlockDiag, Diagonal, HStack, Restriction
from pylops.signalprocessing.sliding2d import _slidingsteps
from pylops.utils._internal import _value_or_list_like_to_tuple
from pylops.utils.tapers import taper

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def sliding1d_design(
    dimd: Tuple,
    nwin: Tuple,
    nover: Tuple,
    nop: Tuple,
) -> Union[Tuple, Tuple, Tuple, Tuple]:
    """Design Sliding1D operator

    This routine can be used prior to creating the :class:`pylops.signalprocessing.Sliding1D`
    operator to identify the correct number of windows to be used based on the dimension of the data (``dimsd``),
    dimension of the window (``nwin``), overlap (``nover``),a and dimension of the operator acting in the model
    space.

    Parameters
    ----------
    dimsd : :obj:`tuple`
        Shape of 2-dimensional data.
    nwin : :obj:`tuple`
        Number of samples of window.
    nover : :obj:`tuple`
        Number of samples of overlapping part of window.
    nop : :obj:`tuple`
        Size of model in the transformed domain.

    Returns
    -------
    nwins : :obj:`int`
        Number of windows.
    dim : :obj:`int`
        Shape of 2-dimensional model.
    mwins_inends : :obj:`tuple`
        Start and end indices for model patches.
    dwins_inends : :obj:`tuple`
        Start and end indices for data patches.

    """
    # data windows
    dwin_ins, dwin_ends = _slidingsteps(dimd, nwin, nover)
    dwins_inends = (dwin_ins, dwin_ends)
    nwins = len(dwin_ins)

    # model windows
    dim = nwins * nop
    mwin_ins, mwin_ends = _slidingsteps(dim, nop, 0)
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
    return nwins, dim, mwins_inends, dwins_inends


def Sliding1D(
    Op,
    dim: Tuple,
    dimd: Tuple,
    nwin: int,
    nover: int,
    tapertype: str = "hanning",
    name: str = "S",
):
    r"""1D Sliding transform operator.

    Apply a transform operator ``Op`` repeatedly to slices of the model
    vector in forward mode and slices of the data vector in adjoint mode.
    More specifically, in forward mode the model vector is divided into
    slices, each slice is transformed, and slices are then recombined in a
    sliding window fashion.

    This operator can be used to perform local, overlapping transforms (e.g.,
    :obj:`pylops.signalprocessing.FFT`) on 1-dimensional arrays.

    .. note:: The shape of the model has to be consistent with
       the number of windows for this operator not to return an error. As the
       number of windows depends directly on the choice of ``nwin`` and
       ``nover``, it is recommended to first run ``sliding1d_design`` to obtain
       the corresponding ``dims`` and number of windows.

    .. warning:: Depending on the choice of `nwin` and `nover` as well as the
       size of the data, sliding windows may not cover the entire data.
       The start and end indices of each window will be displayed and returned
       with running ``sliding1d_design``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Transform operator
    dim : :obj:`tuple`
        Shape of 1-dimensional model.
    dimd : :obj:`tuple`
        Shape of 1-dimensional data
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
    dim = _value_or_list_like_to_tuple(dim)
    dimd = _value_or_list_like_to_tuple(dimd)

    # data windows
    dwin_ins, dwin_ends = _slidingsteps(dimd[0], nwin, nover)
    nwins = len(dwin_ins)

    # check windows
    if nwins * Op.shape[1] != dim[0]:
        raise ValueError(
            f"Model shape (dim={dim}) is not consistent with chosen "
            f"number of windows. Run sliding1d_design to identify the "
            f"correct number of windows for the current "
            "model size..."
        )

    # create tapers
    if tapertype is not None:
        tap = taper(nwin, nover, tapertype=tapertype)
        tapin = tap.copy()
        tapin[:nover] = 1
        tapend = tap.copy()
        tapend[-nover:] = 1
        taps = {}
        taps[0] = tapin
        for i in range(1, nwins - 1):
            taps[i] = tap
        taps[nwins - 1] = tapend

    # transform to apply
    if tapertype is None:
        OOp = BlockDiag([Op for _ in range(nwins)])
    else:
        OOp = BlockDiag([Diagonal(taps[itap].ravel()) * Op for itap in range(nwins)])

    combining = HStack(
        [
            Restriction(dimd, np.arange(win_in, win_end), dtype=Op.dtype).H
            for win_in, win_end in zip(dwin_ins, dwin_ends)
        ]
    )
    Sop = aslinearoperator(combining * OOp)
    Sop.dims, Sop.dimsd = (nwins, int(dim[0] // nwins)), dimd
    Sop.name = name
    return Sop
