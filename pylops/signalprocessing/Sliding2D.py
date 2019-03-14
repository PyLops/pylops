import logging
import numpy as np

from pylops.basicoperators import Diagonal, BlockDiag, Restriction, \
    HStack, VStack
from pylops.utils.tapers import taper2d

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def _slidingsteps(ntr, nwin, nover):
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
        raise ValueError('nwin=%d is bigger than ntr=%d...'
                         % (nwin, ntr))
    step = nwin - nover
    starts = np.arange(0, ntr - nwin + 1, step, dtype=np.int)
    ends = starts + nwin
    return starts, ends


def Sliding2D(Op, dims, dimsd, nwin, nover,
              tapertype='hanning', design=False):
    """Sliding transform operator.

    Apply a transform operator ``Op`` repeatedly to patches of the model
    vector in forward mode and patches of the data vector in adjoint mode.
    More specifically, in forward mode the model vector is divided into patches
    each patch is transformed, and patches are then recombined in a sliding
    window fashion. Both model and data should be  2-dimensional
    arrays in nature as they are internally reshaped and interpreted as
    2-dimensional arrays. Each patch contains in fact a portion of the
    array in the first dimension (and the entire second dimension).

    This operator can be used to perform local, overlapping transforms (e.g.,
    :obj:`pylops.signalprocessing.FFT2`
    or :obj:`pylops.signalprocessing.Radon2D`) of 2-dimensional arrays.

    .. note:: The shape of the model has to be consistent with
       the number of windows for this operator not to return an error. As the
       number of windows depends directly on the choice of ``nwin`` and
       ``nover``, it is recommended to use ``design=True`` if unsure about the
       choice ``dims`` and use the number of windows printed on screen to
       define such input parameter.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Transform operator
    dims : :obj:`tuple`
        Shape of 2-dimensional model. Note that ``dims[0]`` should be multiple
        of the model size of the transform in the first dimension
    dimsd : :obj:`tuple`
        Shape of 2-dimensional data
    nwin : :obj:`tuple`
        Number of samples of window
    nover : :obj:`int`
        Number of samples of overlapping part of window
    tapertype : :obj:`str`, optional
        Type of taper (``hanning``, ``cosine``, ``cosinesquare`` or ``None``)
    design : :obj:`bool`, optional
        Print number sliding window (``True``) or not (``False``)

    Returns
    -------
    Sop : :obj:`pylops.LinearOperator`
        Sliding operator (``None`` if ``design=True``)

    Raises
    ------
    ValueError
        Identified number of windows is not consistent with provided model
        shape (``dims``).

    """
    if tapertype is not None:
        tap = taper2d(dimsd[1], nwin, nover, tapertype=tapertype)
    # model windows
    mwin_ins, mwin_ends = _slidingsteps(dims[0], Op.shape[1]//dims[1], 0)
    # data windows
    dwin_ins, dwin_ends = _slidingsteps(dimsd[0], nwin, nover)
    nwins = len(dwin_ins)

    # check that identified number of windows agrees with mode size
    if design:
        logging.warning('%d windows required...' % nwins)
        logging.warning('model wins - start:%s, end:%s'
                        % (str(mwin_ins), str(mwin_ends)))
        logging.warning('data wins - start:%s, end:%s'
                        % (str(dwin_ins), str(dwin_ends)))
    if nwins*Op.shape[1]//dims[1] != dims[0]:
        raise ValueError('Model shape (dims=%s) is not consistent with chosen '
                         'number of windows. Choose dims[0]=%d for the '
                         'operator to work with estimated number of windows...'
                         % (str(dims), nwins*Op.shape[1]//dims[1]))
    # transform to apply
    if tapertype is None:
        OOp = BlockDiag([Op for _ in range(nwins)])
    else:
        OOp = BlockDiag([Diagonal(tap.flatten()) * Op
                         for _ in range(nwins)])

    slicing = VStack([Restriction(np.prod(dims), range(win_in, win_end),
                                  dims=dims)
                      for win_in, win_end in zip(mwin_ins, mwin_ends)])
    combining = HStack([Restriction(np.prod(dimsd), range(win_in, win_end),
                                    dims=dimsd).H
                        for win_in, win_end in zip(dwin_ins, dwin_ends)])
    Sop = combining * OOp * slicing
    return Sop
