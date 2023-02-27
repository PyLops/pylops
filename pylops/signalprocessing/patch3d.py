__all__ = [
    "patch3d_design",
    "Patch3D",
]

import logging
from typing import Optional, Sequence, Tuple

import numpy as np

from pylops import LinearOperator
from pylops.basicoperators import BlockDiag, Diagonal, HStack, Restriction
from pylops.signalprocessing.sliding2d import _slidingsteps
from pylops.utils.tapers import tapernd
from pylops.utils.typing import InputDimsLike, NDArray

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def patch3d_design(
    dimsd: InputDimsLike,
    nwin: Tuple[int, int, int],
    nover: Tuple[int, int, int],
    nop: Tuple[int, int, int],
) -> Tuple[
    Tuple[int, int, int],
    Tuple[int, int, int],
    Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]],
    Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]],
]:
    """Design Patch3D operator

    This routine can be used prior to creating the :class:`pylops.signalprocessing.Patch3D`
    operator to identify the correct number of windows to be used based on the dimension of the data (``dimsd``),
    dimension of the window (``nwin``), overlap (``nover``),a and dimension of the operator acting in the model
    space.

    Parameters
    ----------
    dimsd : :obj:`tuple`
        Shape of 3-dimensional data.
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
        Shape of 3-dimensional model.
    mwins_inends : :obj:`tuple`
        Start and end indices for model patches (stored as tuple of tuples).
    dwins_inends : :obj:`tuple`
        Start and end indices for data patches (stored as tuple of tuples).

    """
    # data windows
    dwin0_ins, dwin0_ends = _slidingsteps(dimsd[0], nwin[0], nover[0])
    dwin1_ins, dwin1_ends = _slidingsteps(dimsd[1], nwin[1], nover[1])
    dwin2_ins, dwin2_ends = _slidingsteps(dimsd[2], nwin[2], nover[2])
    dwins_inends = (
        (dwin0_ins, dwin0_ends),
        (dwin1_ins, dwin1_ends),
        (dwin2_ins, dwin2_ends),
    )
    nwins0 = len(dwin0_ins)
    nwins1 = len(dwin1_ins)
    nwins2 = len(dwin2_ins)
    nwins = (nwins0, nwins1, nwins2)

    # model windows
    dims = (nwins0 * nop[0], nwins1 * nop[1], nwins2 * nop[2])
    mwin0_ins, mwin0_ends = _slidingsteps(dims[0], nop[0], 0)
    mwin1_ins, mwin1_ends = _slidingsteps(dims[1], nop[1], 0)
    mwin2_ins, mwin2_ends = _slidingsteps(dims[2], nop[2], 0)
    mwins_inends = (
        (mwin0_ins, mwin0_ends),
        (mwin1_ins, mwin1_ends),
        (mwin2_ins, mwin2_ends),
    )

    # print information about patching
    logging.warning("%d-%d-%d windows required...", nwins0, nwins1, nwins2)
    logging.warning(
        "data wins - start:%s, end:%s / start:%s, end:%s / start:%s, end:%s",
        dwin0_ins,
        dwin0_ends,
        dwin1_ins,
        dwin1_ends,
        dwin2_ins,
        dwin2_ends,
    )
    logging.warning(
        "model wins - start:%s, end:%s / start:%s, end:%s / start:%s, end:%s",
        mwin0_ins,
        mwin0_ends,
        mwin1_ins,
        mwin1_ends,
        mwin2_ins,
        mwin2_ends,
    )
    return nwins, dims, mwins_inends, dwins_inends


def Patch3D(
    Op,
    dims: InputDimsLike,
    dimsd: InputDimsLike,
    nwin: Tuple[int, int, int],
    nover: Tuple[int, int, int],
    nop: Tuple[int, int, int],
    tapertype: str = "hanning",
    scalings: Optional[Sequence[float]] = None,
    name: str = "P",
) -> LinearOperator:
    """3D Patch transform operator.

    Apply a transform operator ``Op`` repeatedly to patches of the model
    vector in forward mode and patches of the data vector in adjoint mode.
    More specifically, in forward mode the model vector is divided into
    patches, each patch is transformed, and patches are then recombined
    together. Both model and data are internally reshaped and
    interpreted as 3-dimensional arrays: each patch contains a portion
    of the array in every axis.

    This operator can be used to perform local, overlapping transforms (e.g.,
    :obj:`pylops.signalprocessing.FFTND`
    or :obj:`pylops.signalprocessing.Radon3D`) on 3-dimensional arrays.

    .. note:: The shape of the model has to be consistent with
       the number of windows for this operator not to return an error. As the
       number of windows depends directly on the choice of ``nwin`` and
       ``nover``, it is recommended to first run ``patch3d_design`` to obtain
       the corresponding ``dims`` and number of windows.

    .. warning:: Depending on the choice of `nwin` and `nover` as well as the
       size of the data, sliding windows may not cover the entire data.
       The start and end indices of each window will be displayed and returned
       with running ``patch3d_design``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Transform operator
    dims : :obj:`tuple`
        Shape of 3-dimensional model. Note that ``dims[0]``, ``dims[1]``
        and ``dims[2]`` should be multiple of the model size of the
        transform in their respective dimensions
    dimsd : :obj:`tuple`
        Shape of 3-dimensional data
    nwin : :obj:`tuple`
        Number of samples of window
    nover : :obj:`tuple`
        Number of samples of overlapping part of window
    nop : :obj:`tuple`
        Size of model in the transformed domain
    tapertype : :obj:`str`, optional
        Type of taper (``hanning``, ``cosine``, ``cosinesquare`` or ``None``)
    scalings : :obj:`tuple` or :obj:`list`, optional
         Set of scalings to apply to each patch. If ``None``, no scale will be
         applied
    name : :obj:`str`, optional
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

    See Also
    --------
    Sliding1D: 1D Sliding transform operator.
    Sliding2D: 2D Sliding transform operator.
    Sliding3D: 3D Sliding transform operator.
    Patch2D: 2D Patching transform operator.

    """
    # data windows
    dwin0_ins, dwin0_ends = _slidingsteps(dimsd[0], nwin[0], nover[0])
    dwin1_ins, dwin1_ends = _slidingsteps(dimsd[1], nwin[1], nover[1])
    dwin2_ins, dwin2_ends = _slidingsteps(dimsd[2], nwin[2], nover[2])
    nwins0 = len(dwin0_ins)
    nwins1 = len(dwin1_ins)
    nwins2 = len(dwin2_ins)
    nwins = nwins0 * nwins1 * nwins2

    # check patching
    if (
        nwins0 * nop[0] != dims[0]
        or nwins1 * nop[1] != dims[1]
        or nwins2 * nop[2] != dims[2]
    ):
        raise ValueError(
            f"Model shape (dims={dims}) is not consistent with chosen "
            f"number of windows. Run patch3d_design to identify the "
            f"correct number of windows for the current "
            "model size..."
        )

    # create tapers
    if tapertype is not None:
        tap = tapernd(nwin, nover, tapertype=tapertype).astype(Op.dtype)
        taps = {itap: tap for itap in range(nwins)}
        # 1, sides
        # topmost tapers
        taptop = tap.copy()
        taptop[: nover[0]] = tap[nwin[0] // 2]
        for itap in range(0, nwins1 * nwins2):
            taps[itap] = taptop
        # bottommost tapers
        tapbottom = tap.copy()
        tapbottom[-nover[0] :] = tap[nwin[0] // 2]
        for itap in range(nwins - nwins1 * nwins2, nwins):
            taps[itap] = tapbottom
        # frontmost tapers
        tapfront = tap.copy()
        tapfront[:, :, : nover[2]] = tap[:, :, nwin[2] // 2][:, :, np.newaxis]
        for itap in range(0, nwins, nwins2):
            taps[itap] = tapfront
        # backmost tapers
        tapback = tap.copy()
        tapback[:, :, -nover[2] :] = tap[:, :, nwin[2] // 2][:, :, np.newaxis]
        for itap in range(nwins2 - 1, nwins, nwins2):
            taps[itap] = tapback
        # leftmost tapers
        tapleft = tap.copy()
        tapleft[:, : nover[1]] = tap[:, nwin[1] // 2][:, np.newaxis, :]
        for itap in range(0, nwins, nwins1 * nwins2):
            for i in range(nwins2):
                taps[itap + i] = tapleft
        # rightmost tapers
        tapright = tap.copy()
        tapright[:, -nover[1] :] = tap[:, nwin[1] // 2][:, np.newaxis, :]
        for itap in range(nwins2 * (nwins1 - 1), nwins, nwins2 * nwins1):
            for i in range(nwins2):
                taps[itap + i] = tapright
        # 2. pillars
        # topleftmost tapers
        taplefttop = tap.copy()
        taplefttop[:, : nover[1]] = tap[:, nwin[1] // 2][:, np.newaxis, :]
        taplefttop[: nover[0]] = taplefttop[nwin[0] // 2]
        for itap in range(nwins2):
            taps[itap] = taplefttop
        # toprightmost tapers
        taprighttop = tap.copy()
        taprighttop[:, -nover[1] :] = tap[:, nwin[1] // 2][:, np.newaxis, :]
        taprighttop[: nover[0]] = taprighttop[nwin[0] // 2]
        for itap in range(nwins2):
            taps[nwins2 * (nwins1 - 1) + itap] = taprighttop
        # topfrontmost tapers
        tapfronttop = tap.copy()
        tapfronttop[:, :, : nover[2]] = tap[:, :, nwin[2] // 2][:, :, np.newaxis]
        tapfronttop[: nover[0]] = tapfronttop[nwin[0] // 2]
        for itap in range(0, nwins1 * nwins2, nwins2):
            taps[itap] = tapfronttop
        # topbackmost tapers
        tapbacktop = tap.copy()
        tapbacktop[:, :, -nover[2] :] = tap[:, :, nwin[2] // 2][:, :, np.newaxis]
        tapbacktop[: nover[0]] = tapbacktop[nwin[0] // 2]
        for itap in range(nwins2 - 1, nwins1 * nwins2, nwins2):
            taps[itap] = tapbacktop
        # bottomleftmost tapers
        tapleftbottom = tap.copy()
        tapleftbottom[:, : nover[1]] = tap[:, nwin[1] // 2][:, np.newaxis, :]
        tapleftbottom[-nover[0] :] = tapleftbottom[nwin[0] // 2]
        for itap in range(nwins2):
            taps[(nwins0 - 1) * nwins1 * nwins2 + itap] = tapleftbottom
        # bottomrightmost tapers
        taprightbottom = tap.copy()
        taprightbottom[:, -nover[1] :] = tap[:, nwin[1] // 2][:, np.newaxis, :]
        taprightbottom[-nover[0] :] = taprightbottom[nwin[0] // 2]
        for itap in range(nwins2):
            taps[
                (nwins0 - 1) * nwins1 * nwins2 + (nwins1 - 1) * nwins2 + itap
            ] = taprightbottom
        # bottomfrontmost tapers
        tapfrontbottom = tap.copy()
        tapfrontbottom[:, :, : nover[2]] = tap[:, :, nwin[2] // 2][:, :, np.newaxis]
        tapfrontbottom[-nover[0] :] = tapfrontbottom[nwin[0] // 2]
        for itap in range(0, nwins1 * nwins2, nwins2):
            taps[(nwins0 - 1) * nwins1 * nwins2 + itap] = tapfrontbottom
        # bottombackmost tapers
        tapbackbottom = tap.copy()
        tapbackbottom[:, :, -nover[2] :] = tap[:, :, nwin[2] // 2][:, :, np.newaxis]
        tapbackbottom[-nover[0] :] = tapbackbottom[nwin[0] // 2]
        for itap in range(0, nwins1 * nwins2, nwins2):
            taps[(nwins0 - 1) * nwins1 * nwins2 + nwins2 + itap - 1] = tapbackbottom
        # leftfrontmost tapers
        tapleftfront = tap.copy()
        tapleftfront[:, : nover[1]] = tap[:, nwin[1] // 2][:, np.newaxis, :]
        tapleftfront[:, :, : nover[2]] = tapleftfront[:, :, nwin[2] // 2][
            :, :, np.newaxis
        ]
        for itap in range(0, nwins, nwins1 * nwins2):
            taps[itap] = tapleftfront
        # rightfrontmost tapers
        taprightfront = tap.copy()
        taprightfront[:, -nover[1] :] = tap[:, nwin[1] // 2][:, np.newaxis, :]
        taprightfront[:, :, : nover[2]] = taprightfront[:, :, nwin[2] // 2][
            :, :, np.newaxis
        ]
        for itap in range(0, nwins, nwins1 * nwins2):
            taps[(nwins1 - 1) * nwins2 + itap] = taprightfront
        # leftbackmost tapers
        tapleftback = tap.copy()
        tapleftback[:, : nover[1]] = tap[:, nwin[1] // 2][:, np.newaxis, :]
        tapleftback[:, :, -nover[2] :] = tapleftback[:, :, nwin[2] // 2][
            :, :, np.newaxis
        ]
        for itap in range(0, nwins, nwins1 * nwins2):
            taps[nwins2 + itap - 1] = tapleftback
        # rightbackmost tapers
        taprightback = tap.copy()
        taprightback[:, -nover[1] :] = tap[:, nwin[1] // 2][:, np.newaxis, :]
        taprightback[:, :, -nover[2] :] = taprightback[:, :, nwin[2] // 2][
            :, :, np.newaxis
        ]
        for itap in range(0, nwins, nwins1 * nwins2):
            taps[(nwins1 - 1) * nwins2 + nwins2 + itap - 1] = taprightback
        # 3. corners
        # lefttopfrontcorner taper
        taplefttop = tap.copy()
        taplefttop[: nover[0]] = tap[nwin[0] // 2]
        taplefttop[:, : nover[1]] = taplefttop[:, nwin[1] // 2][:, np.newaxis, :]
        taplefttop[:, :, : nover[2]] = taplefttop[:, :, nwin[2] // 2][:, :, np.newaxis]
        taps[0] = taplefttop
        # lefttopbackcorner taper
        taplefttop = tap.copy()
        taplefttop[: nover[0]] = tap[nwin[0] // 2]
        taplefttop[:, : nover[1]] = taplefttop[:, nwin[1] // 2][:, np.newaxis, :]
        taplefttop[:, :, -nover[2] :] = taplefttop[:, :, nwin[2] // 2][:, :, np.newaxis]
        taps[nwins2 - 1] = taplefttop
        # righttopfrontcorner taper
        taprighttop = tap.copy()
        taprighttop[: nover[0]] = tap[nwin[0] // 2]
        taprighttop[:, -nover[1] :] = taprighttop[:, nwin[1] // 2][:, np.newaxis, :]
        taprighttop[:, :, : nover[2]] = taprighttop[:, :, nwin[2] // 2][
            :, :, np.newaxis
        ]
        taps[(nwins1 - 1) * nwins2] = taprighttop
        # righttopbackcorner taper
        taprighttop = tap.copy()
        taprighttop[: nover[0]] = tap[nwin[0] // 2]
        taprighttop[:, -nover[1] :] = taprighttop[:, nwin[1] // 2][:, np.newaxis, :]
        taprighttop[:, :, -nover[2] :] = taprighttop[:, :, nwin[2] // 2][
            :, :, np.newaxis
        ]
        taps[(nwins1 - 1) * nwins2 + nwins2 - 1] = taprighttop
        # leftbottomfrontcorner taper
        tapleftbottom = tap.copy()
        tapleftbottom[-nover[0] :] = tap[nwin[0] // 2]
        tapleftbottom[:, : nover[1]] = tapleftbottom[:, nwin[1] // 2][:, np.newaxis, :]
        tapleftbottom[:, :, : nover[2]] = tapleftbottom[:, :, nwin[2] // 2][
            :, :, np.newaxis
        ]
        taps[(nwins0 - 1) * nwins1 * nwins2] = tapleftbottom
        # leftbottombackcorner taper
        tapleftbottom = tap.copy()
        tapleftbottom[-nover[0] :] = tap[nwin[0] // 2]
        tapleftbottom[:, : nover[1]] = tapleftbottom[:, nwin[1] // 2][:, np.newaxis, :]
        tapleftbottom[:, :, -nover[2] :] = tapleftbottom[:, :, nwin[2] // 2][
            :, :, np.newaxis
        ]
        taps[(nwins0 - 1) * nwins1 * nwins2 + nwins2 - 1] = tapleftbottom
        # rightbottomfrontcorner taper
        taprightbottom = tap.copy()
        taprightbottom[-nover[0] :] = tap[nwin[0] // 2]
        taprightbottom[:, -nover[1] :] = taprightbottom[:, nwin[1] // 2][
            :, np.newaxis, :
        ]
        taprightbottom[:, :, : nover[2]] = taprightbottom[:, :, nwin[2] // 2][
            :, :, np.newaxis
        ]
        taps[(nwins0 - 1) * nwins1 * nwins2 + (nwins1 - 1) * nwins2] = taprightbottom
        # rightbottombackcorner taper
        taprightbottom = tap.copy()
        taprightbottom[-nover[0] :] = tap[nwin[0] // 2]
        taprightbottom[:, -nover[1] :] = taprightbottom[:, nwin[1] // 2][
            :, np.newaxis, :
        ]
        taprightbottom[:, :, -nover[2] :] = taprightbottom[:, :, nwin[2] // 2][
            :, :, np.newaxis
        ]
        taps[
            (nwins0 - 1) * nwins1 * nwins2 + (nwins1 - 1) * nwins2 + nwins2 - 1
        ] = taprightbottom

    # define scalings
    if scalings is None:
        scalings = [1.0] * nwins

    # transform to apply
    if tapertype is None:
        OOp = BlockDiag([scalings[itap] * Op for itap in range(nwins)])
    else:
        OOp = BlockDiag(
            [
                scalings[itap] * Diagonal(taps[itap].ravel(), dtype=Op.dtype) * Op
                for itap in range(nwins)
            ]
        )

    hstack2 = HStack(
        [
            Restriction(
                (nwin[0], nwin[1], dimsd[2]),
                range(win_in, win_end),
                axis=2,
                dtype=Op.dtype,
            ).H
            for win_in, win_end in zip(dwin2_ins, dwin2_ends)
        ]
    )
    combining2 = BlockDiag([hstack2] * (nwins1 * nwins0))

    hstack1 = HStack(
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
    combining1 = BlockDiag([hstack1] * nwins0)

    combining0 = HStack(
        [
            Restriction(dimsd, range(win_in, win_end), axis=0, dtype=Op.dtype).H
            for win_in, win_end in zip(dwin0_ins, dwin0_ends)
        ]
    )

    Pop = LinearOperator(combining0 * combining1 * combining2 * OOp)
    Pop.dims, Pop.dimsd = (
        nwins0,
        nwins1,
        nwins2,
        int(dims[0] // nwins0),
        int(dims[1] // nwins1),
        int(dims[2] // nwins2),
    ), dimsd
    Pop.name = name
    return Pop
