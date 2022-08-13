import logging

import numpy as np

from pylops.basicoperators import BlockDiag, Diagonal, HStack, Restriction
from pylops.LinearOperator import aslinearoperator
from pylops.signalprocessing.Sliding2D import _slidingsteps
from pylops.utils.tapers import taper2d

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def patch2d_design(dimsd, nwin, nover, nop):
    """Design Patch2D operator

    This routine can be used prior to creating the :class:`pylops.signalprocessing.Patch2D`
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
    dims = (nwins0 * nop[0], nwins1 * nop[1])
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


def Patch2D(
    Op,
    dims,
    dimsd,
    nwin,
    nover,
    nop,
    tapertype="hanning",
    scalings=None,
    name="P",
):
    """2D Patch transform operator.

    Apply a transform operator ``Op`` repeatedly to patches of the model
    vector in forward mode and patches of the data vector in adjoint mode.
    More specifically, in forward mode the model vector is divided into
    patches, each patch is transformed, and patches are then recombined
    together. Both model and data are internally reshaped and
    interpreted as 2-dimensional arrays: each patch contains a portion
    of the array in both the first and second dimension.

    This operator can be used to perform local, overlapping transforms (e.g.,
    :obj:`pylops.signalprocessing.FFT2D`
    or :obj:`pylops.signalprocessing.Radon2D`) on 2-dimensional arrays.

    .. note:: The shape of the model has to be consistent with
       the number of windows for this operator not to return an error. As the
       number of windows depends directly on the choice of ``nwin`` and
       ``nover``, it is recommended to first run ``patch2d_design`` to obtain
       the corresponding ``dims`` and number of windows.

    .. warning:: Depending on the choice of `nwin` and `nover` as well as the
       size of the data, sliding windows may not cover the entire data.
       The start and end indices of each window will be displayed and returned
       with running ``patch2d_design``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Transform operator
    dims : :obj:`tuple`
        Shape of 2-dimensional model. Note that ``dims[0]`` and ``dims[1]``
        should be multiple of the model size of the transform in their
        respective dimensions
    dimsd : :obj:`tuple`
        Shape of 2-dimensional data
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

    See Also
    --------
    Sliding1D: 1D Sliding transform operator.
    Sliding2D: 2D Sliding transform operator.
    Sliding3D: 3D Sliding transform operator.
    Patch3D: 3D Patching transform operator.

    """
    # data windows
    dwin0_ins, dwin0_ends = _slidingsteps(dimsd[0], nwin[0], nover[0])
    dwin1_ins, dwin1_ends = _slidingsteps(dimsd[1], nwin[1], nover[1])
    nwins0 = len(dwin0_ins)
    nwins1 = len(dwin1_ins)
    nwins = nwins0 * nwins1

    # check patching
    if nwins0 * nop[0] != dims[0] or nwins1 * nop[1] != dims[1]:
        raise ValueError(
            f"Model shape (dims={dims}) is not consistent with chosen "
            f"number of windows. Run patch2d_design to identify the "
            f"correct number of windows for the current "
            "model size..."
        )

    # create tapers
    if tapertype is not None:
        tap = taper2d(nwin[1], nwin[0], nover, tapertype=tapertype).astype(Op.dtype)
        taps = {itap: tap for itap in range(nwins)}
        # topmost tapers
        taptop = tap.copy()
        taptop[: nover[0]] = tap[nwin[0] // 2]
        for itap in range(0, nwins1):
            taps[itap] = taptop
        # bottommost tapers
        tapbottom = tap.copy()
        tapbottom[-nover[0] :] = tap[nwin[0] // 2]
        for itap in range(nwins - nwins1, nwins):
            taps[itap] = tapbottom
        # leftmost tapers
        tapleft = tap.copy()
        tapleft[:, : nover[1]] = tap[:, nwin[1] // 2][:, np.newaxis]
        for itap in range(0, nwins, nwins1):
            taps[itap] = tapleft
        # rightmost tapers
        tapright = tap.copy()
        tapright[:, -nover[1] :] = tap[:, nwin[1] // 2][:, np.newaxis]
        for itap in range(nwins1 - 1, nwins, nwins1):
            taps[itap] = tapright
        # lefttopcorner taper
        taplefttop = tap.copy()
        taplefttop[:, : nover[1]] = tap[:, nwin[1] // 2][:, np.newaxis]
        taplefttop[: nover[0]] = taplefttop[nwin[0] // 2]
        taps[0] = taplefttop
        # righttopcorner taper
        taprighttop = tap.copy()
        taprighttop[:, -nover[1] :] = tap[:, nwin[1] // 2][:, np.newaxis]
        taprighttop[: nover[0]] = taprighttop[nwin[0] // 2]
        taps[nwins1 - 1] = taprighttop
        # leftbottomcorner taper
        tapleftbottom = tap.copy()
        tapleftbottom[:, : nover[1]] = tap[:, nwin[1] // 2][:, np.newaxis]
        tapleftbottom[-nover[0] :] = tapleftbottom[nwin[0] // 2]
        taps[nwins - nwins1] = tapleftbottom
        # rightbottomcorner taper
        taprightbottom = tap.copy()
        taprightbottom[:, -nover[1] :] = tap[:, nwin[1] // 2][:, np.newaxis]
        taprightbottom[-nover[0] :] = taprightbottom[nwin[0] // 2]
        taps[nwins - 1] = taprightbottom

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

    hstack = HStack(
        [
            Restriction(
                (nwin[0], dimsd[1]), range(win_in, win_end), axis=1, dtype=Op.dtype
            ).H
            for win_in, win_end in zip(dwin1_ins, dwin1_ends)
        ]
    )
    combining1 = BlockDiag([hstack] * nwins0)

    combining0 = HStack(
        [
            Restriction(dimsd, range(win_in, win_end), axis=0, dtype=Op.dtype).H
            for win_in, win_end in zip(dwin0_ins, dwin0_ends)
        ]
    )
    Pop = aslinearoperator(combining0 * combining1 * OOp)
    Pop.dims, Pop.dimsd = (
        nwins0,
        nwins1,
        int(dims[0] // nwins0),
        int(dims[1] // nwins1),
    ), dimsd
    Pop.name = name
    return Pop
