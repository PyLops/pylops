__all__ = [
    "sliding1d_design",
    "sliding1d_design",
    "Sliding1D",
]

import logging

import numpy as np

from pylops import LinearOperator
from pylops.signalprocessing.sliding2d import _slidingsteps
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import (
    get_array_module,
    get_sliding_window_view,
    to_cupy_conditional,
)
from pylops.utils.decorators import reshaped
from pylops.utils.tapers import taper
from pylops.utils.typing import InputDimsLike, NDArray, Ttaper

logger = logging.getLogger(__name__)


def sliding1d_design(
    dimd: int,
    nwin: int,
    nover: int,
    nop: int,
    verb: bool = True,
) -> tuple[int, int, tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
    """Design Sliding1D operator

    This routine can be used prior to creating the :class:`pylops.signalprocessing.Sliding1D`
    operator to identify the correct number of windows to be used based on the dimension of the data (``dimsd``),
    dimension of the window (``nwin``), overlap (``nover``),a and dimension of the operator acting in the model
    space.

    Parameters
    ----------
    dimd : :obj:`int`
        Shape of the 1-dimensional data.
    nwin : :obj:`int`
        Number of samples of window.
    nover : :obj:`int`
        Number of samples of overlapping part of window.
    nop : :obj:`int`
        Size of model in the transformed domain.
    verb : :obj:`bool`, optional
        *Deprecated*, will be removed in v3.0.0. Simply kept for
        back-compatibility with previous implementation

    Returns
    -------
    nwins : :obj:`int`
        Number of windows.
    dim : :obj:`int`
        Shape of the 1-dimensional model.
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
    logger.info("%d windows required...", nwins)
    logger.info(
        "Data wins - start:%s, end:%s",
        dwin_ins,
        dwin_ends,
    )
    logger.info(
        "Model wins - start:%s, end:%s",
        mwin_ins,
        mwin_ends,
    )
    return nwins, dim, mwins_inends, dwins_inends


def sliding1d_pad_to_next(
    inpt: NDArray,
    nwin: int,
    nover: int,
    nop: int,
) -> tuple[NDArray, int, int, tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
    """Pad input to next slice

    Pad ``inpt`` to the next slice such that the padded input is completely
    filled by overlapping slices.

    Parameters
    ----------
    inpt : :obj:`numpy.ndarray`
        1-dimensional input data.
    nwin : :obj:`tuple`
        Number of samples of window.
    nover : :obj:`int`
        Number of samples of overlapping part of window.
    nop : :obj:`int`
        Size of model in the transformed domain.

    Returns
    -------
    inpt_pad : :obj:`numpy.ndarray`
        1-dimensional input data after padding
    nwins : :obj:`int`
        Number of windows of padded input.
    dim : :obj:`int`
        Shape of the 1-dimensional model of padded input.
    mwins_inends : :obj:`tuple`
        Start and end indices for model patches of padded input.
    dwins_inends : :obj:`tuple`
        Start and end indices for data patches of padded input.

    """
    # Identify current sliding design
    dimd = inpt.size
    nwins, dim, mwins_inends, dwins_inends = sliding1d_design(dimd, nwin, nover, nop)

    # Pad to next slice
    if dwins_inends[1][-1] != dimd:
        pad = dwins_inends[1][-1] - nover + nwin - dimd
        inpt_pad = np.pad(inpt, (0, pad))
        dimd_pad = inpt_pad.size
        nwins, dim, mwins_inends, dwins_inends = sliding1d_design(
            dimd_pad, nwin, nover, nop
        )
    else:
        inpt_pad = inpt
    return inpt_pad, nwins, dim, mwins_inends, dwins_inends


class Sliding1D(LinearOperator):
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

    .. note:: Two kind of operators ``Op`` can be provided: the first
       applies a single transformation to each window separately; the second
       applies the transformation to all of the windows at the same time. This
       is directly inferred during initialization when the following condition
       holds ``Op.shape[1] == dim[0]``.

    .. warning:: Depending on the choice of `nwin` and `nover` as well as the
       size of the data, sliding windows may not cover the entire data.
       The start and end indices of each window will be displayed and returned
       with running ``sliding1d_design``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Transform operator
    dim : :obj:`tuple`
        Shape of the 1-dimensional model
    dimd : :obj:`tuple`
        Shape of the 1-dimensional data
    nwin : :obj:`int`
        Number of samples of window
    nover : :obj:`int`
        Number of samples of overlapping part of window
    tapertype : :obj:`str`, optional
        Type of taper (``hanning``, ``cosine``, ``cosinesquare`` or ``None``)
    savetaper : :obj:`bool`, optional
        .. versionadded:: 2.3.0

        Save all tapers and apply them in one go (``True``) or save unique tapers and apply them one by one (``False``).
        The first option is more computationally efficient, whilst the second is more memory efficient.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    taps : :obj:`numpy.ndarray`
        Set of tapers applied to each window (only if ``tapertype`` is not ``None``)
    simOp : :obj:`bool`
        Operator ``Op`` is applied to all windows simultaneously (``True``)
        or to each window individually (``False``)
    dims : :obj:`tuple`
        Shape of the array after the adjoint, but before flattening.

        For example, ``x_reshaped = (Op.H * y.ravel()).reshape(Op.dims)``.
    dimsd : :obj:`tuple`
        Shape of the array after the forward, but before flattening.

        For example, ``y_reshaped = (Op * x.ravel()).reshape(Op.dimsd)``.
    shape : :obj:`tuple`
        Operator shape.

    Raises
    ------
    ValueError
        Identified number of windows is not consistent with provided model
        shape (``dims``).

    """

    def __init__(
        self,
        Op: LinearOperator,
        dim: int | InputDimsLike,
        dimd: int | InputDimsLike,
        nwin: int,
        nover: int,
        tapertype: Ttaper | None = "hanning",
        savetaper: bool = True,
        name: str = "S",
    ) -> None:
        dim: tuple[int, ...] = _value_or_sized_to_tuple(dim)
        dimd: tuple[int, ...] = _value_or_sized_to_tuple(dimd)

        # data windows
        dwin_ins, dwin_ends = _slidingsteps(dimd[0], nwin, nover)
        self.dwin_inends = (dwin_ins, dwin_ends)
        nwins = len(dwin_ins)
        self.nwin = nwin
        self.nover = nover

        # check windows
        if nwins * Op.shape[1] != dim[0] and Op.shape[1] != dim[0]:
            msg = (
                f"Model shape (dim={dim}) is not consistent with chosen number of windows. "
                "Run sliding1d_design to identify the correct number of windows for the current model size..."
            )
            raise ValueError(msg)

        # create tapers
        self.tapertype = tapertype
        self.savetaper = savetaper
        if self.tapertype is not None:
            tap = taper(nwin, nover, tapertype=self.tapertype).astype(Op.dtype)
            tapin = tap.copy()
            tapin[:nover] = 1
            tapend = tap.copy()
            tapend[-nover:] = 1
            if self.savetaper:
                self.taps = [
                    tapin,
                ]
                for _ in range(1, nwins - 1):
                    self.taps.append(tap)
                self.taps.append(tapend)
                self.taps = np.vstack(self.taps)
            else:
                self.taps = np.vstack([tapin, tap, tapend])

        # check if operator is applied to all windows simultaneously
        self.simOp = False
        if Op.shape[1] == dim[0]:
            self.simOp = True
        self.Op = Op

        super().__init__(
            dtype=Op.dtype,
            dims=(nwins, int(dim[0] // nwins)),
            dimsd=dimd,
            clinear=False,
            name=name,
        )

        self._register_multiplications(self.savetaper)

    def _apply_taper(self, ywins, iwin0):
        if iwin0 == 0:
            ywins[0] = ywins[0] * self.taps[0]
        elif iwin0 == self.dims[0] - 1:
            ywins[-1] = ywins[-1] * self.taps[-1]
        else:
            ywins[iwin0] = ywins[iwin0] * self.taps[1]
        return ywins

    @reshaped
    def _matvec_savetaper(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if self.tapertype is not None:
            self.taps = to_cupy_conditional(x, self.taps)
        y = ncp.zeros(self.dimsd, dtype=self.dtype)
        if self.simOp:
            x = self.Op.matvec(x.ravel()).reshape(self.Op.dimsd)
            if self.tapertype is not None:
                x = self.taps * x
        for iwin0 in range(self.dims[0]):
            if self.simOp:
                xxwin = x[iwin0]
            else:
                xxwin = self.Op.matvec(x[iwin0])
                if self.tapertype is not None:
                    xxwin = self.taps[iwin0] * xxwin
            y[self.dwin_inends[0][iwin0] : self.dwin_inends[1][iwin0]] += xxwin
        return y

    @reshaped
    def _rmatvec_savetaper(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        ncp_sliding_window_view = get_sliding_window_view(x)
        if self.tapertype is not None:
            self.taps = to_cupy_conditional(x, self.taps)
        ywins = ncp_sliding_window_view(x, self.nwin)[:: self.nwin - self.nover]
        if self.tapertype is not None:
            ywins = ywins * self.taps
        if self.simOp:
            y = self.Op.rmatvec(ywins.ravel()).reshape(self.dims)
        else:
            y = ncp.zeros(self.dims, dtype=self.dtype)
            for iwin0 in range(self.dims[0]):
                y[iwin0] = self.Op.rmatvec(ywins[iwin0])
        return y

    @reshaped
    def _matvec_nosavetaper(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if self.tapertype is not None:
            self.taps = to_cupy_conditional(x, self.taps)
        y = ncp.zeros(self.dimsd, dtype=self.dtype)
        if self.simOp:
            x = self.Op.matvec(x.ravel()).reshape(self.Op.dimsd)
        for iwin0 in range(self.dims[0]):
            if self.simOp:
                xxwin = x[iwin0]
            else:
                xxwin = self.Op.matvec(x[iwin0])
            if self.tapertype is not None:
                if iwin0 == 0:
                    xxwin = self.taps[0] * xxwin
                elif iwin0 == self.dims[0] - 1:
                    xxwin = self.taps[-1] * xxwin
                else:
                    xxwin = self.taps[1] * xxwin
            y[self.dwin_inends[0][iwin0] : self.dwin_inends[1][iwin0]] += xxwin
        return y

    @reshaped
    def _rmatvec_nosavetaper(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        ncp_sliding_window_view = get_sliding_window_view(x)
        if self.tapertype is not None:
            self.taps = to_cupy_conditional(x, self.taps)
        ywins = ncp_sliding_window_view(x, self.nwin)[:: self.nwin - self.nover].copy()
        if self.simOp:
            if self.tapertype is not None:
                for iwin0 in range(self.dims[0]):
                    ywins = self._apply_taper(ywins, iwin0)
            y = self.Op.rmatvec(ywins.ravel())
        else:
            y = ncp.zeros(self.dims, dtype=self.dtype)
            for iwin0 in range(self.dims[0]):
                if self.tapertype is not None:
                    ywins = self._apply_taper(ywins, iwin0)
                y[iwin0] = self.Op.rmatvec(ywins[iwin0])
        return y

    def _register_multiplications(self, savetaper: bool) -> None:
        if savetaper:
            self._matvec = self._matvec_savetaper
            self._rmatvec = self._rmatvec_savetaper
        else:
            self._matvec = self._matvec_nosavetaper
            self._rmatvec = self._rmatvec_nosavetaper
