import logging
import numpy as np

from pylops.basicoperators import Diagonal, BlockDiag, Restriction, \
    HStack, VStack
from pylops.utils.tapers import taper2d

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def _slidingsteps(ntr, nwin, nover):
    """Identify sliding window initial and end points given trace lenght,
    window lenght and overlap

    Parameters
    ----------
    ntr : :obj:`int`
        Number of samples in trace
    nwin : :obj:`int`
        Number of samples of window
    nover : :obj:`int`
        Number of samples of overlapping part of taper

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

    Applies a transform operator ``Op`` repeatedly to patches of the model
    vector in forward mode and patches of the data vector in adjoint mode.
    In forward mode, the model vector is divided into patches and each
    patch is transformed, patches are then recombined in a sliding window
    fashion. Both model and data are should be in nature 2-dimensional
    arrays as they are internally reshaped and interpreted as 2-dimensional
    arrays and divided into patches. Each patch contains a portion of the
    array in the first dimension (and the entire second dimension).

    This operator can be used to perform local, overlapping transforms (e.g.,
    :obj:`pylops.signalprocessing.FFT2`
    or :obj:`pylops.signalprocessing.Radon2D`) of 2-dimensional arrays.

    .. note:: The shape of the model has to be consistent with
       the number of windows for this operator not to return an error. As the
       number of windows depends directly on the choice of ``nwin`` and
       ``nover``, it is reccomended to first create the operator in design mode
       (``design=True``) to find out the number of windows required.

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
        Number of samples of overlapping part of taper
    tapertype : :obj:`str`, optional
        Type of taper (``hanning``, ``cosine``, ``cosinesquare`` or ``None``)
    design : :obj:`bool`, optional
        Design sliding window programme (``True``) or initialize operator
        (``False``)

    Returns
    -------
    Sop : :obj:`pylops.LinearOperator`
        Sliding operator (``None`` if ``design=True``)

    Raises
    ------
    ValueError
        If ``dims[0]`` is not multiple of the model size of the transform ``Op``
        in the first dimension.

    """
    if dims[0] % (Op.shape[1]/dims[1]) > 0:
        raise ValueError('dims[0]={} is not multiple of the size of '
                         'the transform in the first '
                         'dimension={}.'.format(dims[0], Op.shape[1]//dims[1]))

    if tapertype is not None:
        tap = taper2d(dimsd[1], nwin, nover, tapertype=tapertype)
    # model windows
    mwin_ins, mwin_ends = _slidingsteps(dims[0], Op.shape[1]//dims[1], 0)
    # data windows
    dwin_ins, dwin_ends = _slidingsteps(dimsd[0], nwin, nover)
    if design:
        logging.warning('%d windows requires...' % len(mwin_ins))
        Sop = None
    else:
        # transform to apply
        if tapertype is None:
            OOp = BlockDiag([Op for _ in range(len(dwin_ins))])
        else:
            OOp = BlockDiag([Diagonal(tap.flatten()) * Op
                             for _ in range(len(mwin_ins))])

        slicing = VStack([Restriction(np.prod(dims), range(win_in, win_end),
                                      dims=dims)
                          for win_in, win_end in zip(mwin_ins, mwin_ends)])
        combining = HStack([Restriction(np.prod(dimsd), range(win_in, win_end),
                                        dims=dimsd).H
                            for win_in, win_end in zip(dwin_ins, dwin_ends)])
        Sop = combining * OOp * slicing
    return Sop
