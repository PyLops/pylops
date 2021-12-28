import logging

import numpy as np

from pylops.basicoperators import BlockDiag, Diagonal, HStack, Restriction
from pylops.signalprocessing.Sliding2D import _slidingsteps
from pylops.utils.tapers import taper2d

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def Patch2D(Op, dims, dimsd, nwin, nover, nop, tapertype="hanning", design=False):
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
       ``nover``, it is recommended to use ``design=True`` if unsure about the
       choice ``dims`` and use the number of windows printed on screen to
       define such input parameter.

    .. warning:: Depending on the choice of `nwin` and `nover` as well as the
       size of the data, patches may not cover the entire size of the data.
       The start and end indices of each window can be displayed using
       ``design=True`` while defining the best patching approach.

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

    See Also
    --------
    Sliding2d: 2D Sliding transform operator.

    """
    # model windows
    mwin0_ins, mwin0_ends = _slidingsteps(dims[0], nop[0], 0)
    mwin1_ins, mwin1_ends = _slidingsteps(dims[1], nop[1], 0)

    # data windows
    dwin0_ins, dwin0_ends = _slidingsteps(dimsd[0], nwin[0], nover[0])
    dwin1_ins, dwin1_ends = _slidingsteps(dimsd[1], nwin[1], nover[1])
    nwins0 = len(dwin0_ins)
    nwins1 = len(dwin1_ins)
    nwins = nwins0 * nwins1

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

    # check that identified number of windows agrees with mode size
    if design:
        logging.warning("%d-%d windows required...", nwins0, nwins1)
        logging.warning(
            "model wins - start:%s, end:%s / start:%s, end:%s",
            str(mwin0_ins),
            str(mwin0_ends),
            str(mwin1_ins),
            str(mwin1_ends),
        )
        logging.warning(
            "data wins - start:%s, end:%s / start:%s, end:%s",
            str(dwin0_ins),
            str(dwin0_ends),
            str(dwin1_ins),
            str(dwin1_ends),
        )
    if nwins0 * nop[0] != dims[0] or nwins1 * nop[1] != dims[1]:
        raise ValueError(
            "Model shape (dims=%s) is not consistent with chosen "
            "number of windows. Choose dims[0]=%d and "
            "dims[1]=%d for the operator to work with "
            "estimated number of windows, or create "
            "the operator with design=True to find out the"
            "optimal number of windows for the current "
            "model size..." % (str(dims), nwins0 * nop[0], nwins1 * nop[1])
        )
    # transform to apply
    if tapertype is None:
        OOp = BlockDiag([Op for _ in range(nwins)])
    else:
        OOp = BlockDiag(
            [Diagonal(taps[itap].ravel(), dtype=Op.dtype) * Op for itap in range(nwins)]
        )

    hstack = HStack(
        [
            Restriction(
                dimsd[1] * nwin[0],
                range(win_in, win_end),
                dims=(nwin[0], dimsd[1]),
                dir=1,
                dtype=Op.dtype,
            ).H
            for win_in, win_end in zip(dwin1_ins, dwin1_ends)
        ]
    )

    combining1 = BlockDiag([hstack] * nwins0)
    combining0 = HStack(
        [
            Restriction(
                np.prod(dimsd),
                range(win_in, win_end),
                dims=dimsd,
                dir=0,
                dtype=Op.dtype,
            ).H
            for win_in, win_end in zip(dwin0_ins, dwin0_ends)
        ]
    )
    Pop = combining0 * combining1 * OOp
    return Pop
