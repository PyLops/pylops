import logging

import numpy as np

from pylops.basicoperators import BlockDiag, Diagonal, HStack, Restriction
from pylops.LinearOperator import aslinearoperator
from pylops.signalprocessing.Sliding2D import _slidingsteps
from pylops.utils.tapers import taper

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def Sliding1D(Op, dim, dimd, nwin, nover, tapertype="hanning", design=False):
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
       ``nover``, it is recommended to use ``design=True`` if unsure about the
       choice ``dims`` and use the number of windows printed on screen to
       define such input parameter.

    .. warning:: Depending on the choice of `nwin` and `nover` as well as the
       size of the data, sliding windows may not cover the entire data.
       The start and end indices of each window can be displayed using
       ``design=True`` while defining the best sliding window approach.

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
    design : :obj:`bool`, optional
        Print number of sliding window (``True``) or not (``False``)

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
    # model windows
    mwin_ins, mwin_ends = _slidingsteps(dim, Op.shape[1], 0)
    # data windows
    dwin_ins, dwin_ends = _slidingsteps(dimd, nwin, nover)
    nwins = len(dwin_ins)

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

    # check that identified number of windows agrees with mode size
    if design:
        logging.warning("%d windows required...", nwins)
        logging.warning("model wins - start:%s, end:%s", mwin_ins, mwin_ends)
        logging.warning("data wins - start:%s, end:%s", dwin_ins, dwin_ends)
    if nwins * Op.shape[1] != dim:
        raise ValueError(
            f"Model shape (dim={dim}) is not consistent with chosen "
            f"number of windows. Choose dim={nwins * Op.shape[1]} for the "
            "operator to work with estimated number of windows, "
            "or create the operator with design=True to find "
            "out the optimal number of windows for the current "
            "model size..."
        )
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
    Sop.dims, Sop.dimsd = (nwins, int(dim // nwins)), dimd
    return Sop
