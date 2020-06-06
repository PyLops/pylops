import logging
import numpy as np

from pylops.basicoperators import Diagonal, BlockDiag, Restriction, \
    HStack
from pylops.utils.tapers import taper3d
from pylops.signalprocessing.Sliding2D import _slidingsteps

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def Sliding3D(Op, dims, dimsd, nwin, nover, nop,
              tapertype='hanning', design=False):
    """3D Sliding transform operator.

    Apply a transform operator ``Op`` repeatedly to patches of the model
    vector in forward mode and patches of the data vector in adjoint mode.
    More specifically, in forward mode the model vector is divided into patches
    each patch is transformed, and patches are then recombined in a sliding
    window fashion. Both model and data should be 3-dimensional
    arrays in nature as they are internally reshaped and interpreted as
    3-dimensional arrays. Each patch contains in fact a portion of the
    array in the first and second dimensions (and the entire third dimension).

    This operator can be used to perform local, overlapping transforms (e.g.,
    :obj:`pylops.signalprocessing.FFTN`
    or :obj:`pylops.signalprocessing.Radon3D`) of 3-dimensional arrays.

    .. note:: The shape of the model has to be consistent with
       the number of windows for this operator not to return an error. As the
       number of windows depends directly on the choice of ``nwin`` and
       ``nover``, it is recommended to use ``design=True`` if unsure about the
       choice ``dims`` and use the number of windows printed on screen to
       define such input parameter.

    .. warning:: Depending on the choice of `nwin` and `nover` as well as the
       size of the data, sliding windows may not cover the entire first and/or
       second dimensions. The start and end indeces of each window can be
       displayed using ``design=True`` while defining the best sliding window
       approach.

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
    design : :obj:`bool`, optional
        Print number sliding window (``True``) or not (``False``)

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
    mwin0_ins, mwin0_ends = _slidingsteps(dims[0],
                                          Op.shape[1]//(nop[1]*dims[2]), 0)
    mwin1_ins, mwin1_ends = _slidingsteps(dims[1],
                                          Op.shape[1]//(nop[0]*dims[2]), 0)

    # data windows
    dwin0_ins, dwin0_ends = _slidingsteps(dimsd[0], nwin[0], nover[0])
    dwin1_ins, dwin1_ends = _slidingsteps(dimsd[1], nwin[1], nover[1])
    nwins0 = len(dwin0_ins)
    nwins1 = len(dwin1_ins)
    nwins = nwins0*nwins1

    # create tapers
    if tapertype is not None:
        tap = taper3d(dimsd[2], nwin, nover, tapertype=tapertype)

    # check that identified number of windows agrees with mode size
    if design:
        logging.warning('(%d,%d) windows required...', nwins0, nwins1)
        logging.warning('model wins - start0:%s, end0:%s, start1:%s, end1:%s',
                        str(mwin0_ins), str(mwin0_ends),
                        str(mwin1_ins), str(mwin1_ends))
        logging.warning('data wins - start0:%s, end0:%s, start1:%s, end1:%s',
                        str(dwin0_ins), str(dwin0_ends),
                        str(dwin1_ins), str(dwin1_ends))

    if nwins*Op.shape[1]//dims[2] != dims[0]*dims[1]:
        raise ValueError('Model shape (dims=%s) is not consistent with chosen '
                         'number of windows. Choose dims[0]=%d and '
                         'dims[1]=%d for the operator to work with '
                         'estimated number of windows, or create '
                         'the operator with design=True to find out the'
                         'optimal number of windows for the current '
                         'model size...'
                         % (str(dims), nwins0*Op.shape[1]//(nop[1]*dims[2]),
                            nwins1 * Op.shape[1]//(nop[0]*dims[2])))
    # transform to apply
    if tapertype is None:
        OOp = BlockDiag([Op for _ in range(nwins)])
    else:
        OOp = BlockDiag([Diagonal(tap.flatten()) * Op
                         for _ in range(nwins)])

    hstack = HStack([Restriction(dimsd[1] * dimsd[2] * nwin[0],
                                 range(win_in, win_end),
                                 dims=(nwin[0], dimsd[1], dimsd[2]),
                                 dir=1).H
                     for win_in, win_end in zip(dwin1_ins,
                                                dwin1_ends)])

    combining1 = BlockDiag([hstack]*nwins0)
    combining0 = HStack([Restriction(np.prod(dimsd),
                                     range(win_in, win_end),
                                     dims=dimsd, dir=0).H
                         for win_in, win_end in zip(dwin0_ins, dwin0_ends)])
    Sop = combining0 * combining1 * OOp
    return Sop
