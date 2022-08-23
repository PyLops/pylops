__all__ = [
    "sliding3d_design",
    "Sliding3D",
]

import logging
from typing import List, Tuple, Union

from pylops import aslinearoperator
from pylops.basicoperators import BlockDiag, Diagonal, HStack, Restriction
from pylops.signalprocessing.sliding2d import _slidingsteps
from pylops.utils.tapers import taper3d

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def sliding3d_design(
    dimsd: Tuple,
    nwin: Tuple,
    nover: Tuple,
    nop: Tuple,
) -> Union[Tuple, Tuple, Tuple, Tuple]:
    """Design Sliding3D operator

    This routine can be used prior to creating the :class:`pylops.signalprocessing.Sliding3D`
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
    nwins : :obj:`tuple`
        Number of windows.
    dims : :obj:`tuple`
        Shape of 2-dimensional model.
    mwins_inends : :obj:`tuple`
        Start and end indices for model patches (stored as tuple of tuples).
    dwins_inends : :obj:`tuple`
        Start and end indices for data patches (stored as tuple of tuples).

    """
    # data windows
    dwin0_ins, dwin0_ends = _slidingsteps(dimsd[0], nwin[0], nover[0])
    dwin1_ins, dwin1_ends = _slidingsteps(dimsd[1], nwin[1], nover[1])
    dwins_inends = ((dwin0_ins, dwin0_ends), (dwin1_ins, dwin1_ends))
    nwins0 = len(dwin0_ins)
    nwins1 = len(dwin1_ins)
    nwins = (nwins0, nwins1)

    # model windows
    dims = (nwins0 * nop[0], nwins1 * nop[1], nop[2])
    mwin0_ins, mwin0_ends = _slidingsteps(dims[0], nop[0], 0)
    mwin1_ins, mwin1_ends = _slidingsteps(dims[1], nop[1], 0)
    mwins_inends = ((mwin0_ins, mwin0_ends), (mwin1_ins, mwin1_ends))

    # print information about patching
    logging.warning("%d-%d windows required...", nwins0, nwins1)
    logging.warning(
        "data wins - start:%s, end:%s / start:%s, end:%s",
        dwin0_ins,
        dwin0_ends,
        dwin1_ins,
        dwin1_ends,
    )
    logging.warning(
        "model wins - start:%s, end:%s / start:%s, end:%s",
        mwin0_ins,
        mwin0_ends,
        mwin1_ins,
        mwin1_ends,
    )
    return nwins, dims, mwins_inends, dwins_inends


def Sliding3D(
    Op,
    dims: Tuple,
    dimsd: Tuple,
    nwin: Tuple,
    nover: Tuple,
    nop: Tuple,
    tapertype: str = "hanning",
    nproc: int = 1,
    name: str = "P",
) -> None:
    """3D Sliding transform operator.w

    Apply a transform operator ``Op`` repeatedly to patches of the model
    vector in forward mode and patches of the data vector in adjoint mode.
    More specifically, in forward mode the model vector is divided into patches
    each patch is transformed, and patches are then recombined in a sliding
    window fashion. Both model and data should be 3-dimensional
    arrays in nature as they are internally reshaped and interpreted as
    3-dimensional arrays. Each patch contains in fact a portion of the
    array in the first and second dimensions (and the entire third dimension).

    This operator can be used to perform local, overlapping transforms (e.g.,
    :obj:`pylops.signalprocessing.FFTND`
    or :obj:`pylops.signalprocessing.Radon3D`) of 3-dimensional arrays.

    .. note:: The shape of the model has to be consistent with
       the number of windows for this operator not to return an error. As the
       number of windows depends directly on the choice of ``nwin`` and
       ``nover``, it is recommended to first run ``sliding3d_design`` to obtain
       the corresponding ``dims`` and number of windows.

    .. warning:: Depending on the choice of `nwin` and `nover` as well as the
       size of the data, sliding windows may not cover the entire data.
       The start and end indices of each window will be displayed and returned
       with running ``sliding3d_design``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Transform operator
    dims : :obj:`tuple`
        Shape of 3-dimensional model. Note that ``dims[0]`` and ``dims[1]``
        should be multiple of the model sizes of the transform in the
        first and second dimensions
    dimsd : :obj:`tuple`
        Shape of 3-dimensional data
    nwin : :obj:`tuple`
        Number of samples of window
    nover : :obj:`tuple`
        Number of samples of overlapping part of window
    nop : :obj:`tuple`
        Number of samples in axes of transformed domain associated
        to spatial axes in the data
    tapertype : :obj:`str`, optional
        Type of taper (``hanning``, ``cosine``, ``cosinesquare`` or ``None``)
    nproc : :obj:`int`, optional
        Number of processes used to evaluate the N operators in parallel
        using ``multiprocessing``. If ``nproc=1``, work in serial mode.
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
    dwin0_ins, dwin0_ends = _slidingsteps(dimsd[0], nwin[0], nover[0])
    dwin1_ins, dwin1_ends = _slidingsteps(dimsd[1], nwin[1], nover[1])
    nwins0 = len(dwin0_ins)
    nwins1 = len(dwin1_ins)
    nwins = nwins0 * nwins1

    # check windows
    if nwins * Op.shape[1] // dims[2] != dims[0] * dims[1]:
        raise ValueError(
            f"Model shape (dims={dims}) is not consistent with chosen "
            f"number of windows. Run sliding3d_design to identify the "
            f"correct number of windows for the current "
            "model size..."
        )

    # create tapers
    if tapertype is not None:
        tap = taper3d(dimsd[2], nwin, nover, tapertype=tapertype)

    # transform to apply
    if tapertype is None:
        OOp = BlockDiag([Op for _ in range(nwins)], nproc=nproc)
    else:
        OOp = BlockDiag([Diagonal(tap.ravel()) * Op for _ in range(nwins)], nproc=nproc)

    hstack = HStack(
        [
            Restriction(
                (nwin[0], dimsd[1], dimsd[2]),
                range(win_in, win_end),
                axis=1,
                dtype=Op.dtype,
            ).H
            for win_in, win_end in zip(dwin1_ins, dwin1_ends)
        ]
    )

    combining1 = BlockDiag([hstack] * nwins0)
    combining0 = HStack(
        [
            Restriction(
                dimsd,
                range(win_in, win_end),
                axis=0,
                dtype=Op.dtype,
            ).H
            for win_in, win_end in zip(dwin0_ins, dwin0_ends)
        ]
    )
    Sop = aslinearoperator(combining0 * combining1 * OOp)
    Sop.dims, Sop.dimsd = (
        nwins0,
        nwins1,
        int(dims[0] // nwins0),
        int(dims[1] // nwins1),
        dims[2],
    ), dimsd
    Sop.name = name
    return Sop
