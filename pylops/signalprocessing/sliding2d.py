__all__ = [
    "sliding2d_design",
    "Sliding2D",
]

import logging
from typing import Tuple

import numpy as np

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.backend import (
    get_array_module,
    get_sliding_window_view,
    to_cupy_conditional,
)
from pylops.utils.decorators import reshaped
from pylops.utils.tapers import taper2d
from pylops.utils.typing import InputDimsLike, NDArray

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def _slidingsteps(
    ntr: int,
    nwin: int,
    nover: int,
) -> Tuple[NDArray, NDArray]:
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
        raise ValueError(f"nwin={nwin} is bigger than ntr={ntr}...")
    step = nwin - nover
    starts = np.arange(0, ntr - nwin + 1, step, dtype=int)
    ends = starts + nwin
    return starts, ends


def sliding2d_design(
    dimsd: Tuple[int, int],
    nwin: int,
    nover: int,
    nop: Tuple[int, int],
    verb: bool = True,
) -> Tuple[int, Tuple[int, int], Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
    """Design Sliding2D operator

    This routine can be used prior to creating the :class:`pylops.signalprocessing.Sliding2D`
    operator to identify the correct number of windows to be used based on the dimension of the data (``dimsd``),
    dimension of the window (``nwin``), overlap (``nover``),a and dimension of the operator acting in the model
    space.

    Parameters
    ----------
    dimsd : :obj:`tuple`
        Shape of 2-dimensional data.
    nwin : :obj:`int`
        Number of samples of window.
    nover : :obj:`int`
        Number of samples of overlapping part of window.
    nop : :obj:`tuple`
        Size of model in the transformed domain.
    verb : :obj:`bool`, optional
        Verbosity flag. If ``verb==True``, print the data
        and model windows start-end indices

    Returns
    -------
    nwins : :obj:`int`
        Number of windows.
    dims : :obj:`tuple`
        Size of 2-dimensional model.
    mwins_inends : :obj:`tuple`
        Start and end indices for model patches (stored as tuple of tuples).
    dwins_inends : :obj:`tuple`
        Start and end indices for data patches (stored as tuple of tuples).

    """
    # data windows
    dwin_ins, dwin_ends = _slidingsteps(dimsd[0], nwin, nover)
    dwins_inends = (dwin_ins, dwin_ends)
    nwins = len(dwin_ins)

    # model windows
    dims = (nwins * nop[0], nop[1])
    mwin_ins, mwin_ends = _slidingsteps(dims[0], nop[0], 0)
    mwins_inends = (mwin_ins, mwin_ends)

    # print information about patching
    if verb:
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
    return nwins, dims, mwins_inends, dwins_inends


class Sliding2D(LinearOperator):
    """2D Sliding transform operator.

    Apply a transform operator ``Op`` repeatedly to slices of the model
    vector in forward mode and slices of the data vector in adjoint mode.
    More specifically, in forward mode the model vector is divided into
    slices, each slice is transformed, and slices are then recombined in a
    sliding window fashion. Both model and data are internally reshaped and
    interpreted as 2-dimensional arrays: each slice contains a portion
    of the array in the first dimension (and the entire second dimension).

    This operator can be used to perform local, overlapping transforms (e.g.,
    :obj:`pylops.signalprocessing.FFT2D`
    or :obj:`pylops.signalprocessing.Radon2D`) on 2-dimensional arrays.

    .. note:: The shape of the model has to be consistent with
       the number of windows for this operator not to return an error. As the
       number of windows depends directly on the choice of ``nwin`` and
       ``nover``, it is recommended to first run ``sliding2d_design`` to obtain
       the corresponding ``dims`` and number of windows.

    .. note:: Two kind of operators ``Op`` can be provided: the first
       applies a single transformation to each window separately; the second
       applies the transformation to all of the windows at the same time. This
       is directly inferred during initialization when the following condition
       holds ``Op.shape[1] == np.prod(dims)``.

    .. warning:: Depending on the choice of `nwin` and `nover` as well as the
       size of the data, sliding windows may not cover the entire data.
       The start and end indices of each window will be displayed and returned
       with running ``sliding2d_design``.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Transform operator
    dims : :obj:`tuple`
        Shape of 2-dimensional model. Note that ``dims[0]`` should be multiple
        of the model size of the transform in the first dimension
    dimsd : :obj:`tuple`
        Shape of 2-dimensional data
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

    def __init__(
        self,
        Op: LinearOperator,
        dims: InputDimsLike,
        dimsd: InputDimsLike,
        nwin: int,
        nover: int,
        tapertype: str = "hanning",
        savetaper: bool = True,
        name: str = "S",
    ) -> None:

        dims: Tuple[int, ...] = _value_or_sized_to_tuple(dims)
        dimsd: Tuple[int, ...] = _value_or_sized_to_tuple(dimsd)

        # data windows
        dwin_ins, dwin_ends = _slidingsteps(dimsd[0], nwin, nover)
        self.dwin_inends = (dwin_ins, dwin_ends)
        nwins = len(dwin_ins)
        self.nwin = nwin
        self.nover = nover

        # check patching
        if nwins * Op.shape[1] // dims[1] != dims[0] and Op.shape[1] != np.prod(dims):
            raise ValueError(
                f"Model shape (dims={dims}) is not consistent with chosen "
                f"number of windows. Run sliding2d_design to identify the "
                f"correct number of windows for the current "
                "model size..."
            )

        # create tapers
        self.tapertype = tapertype
        self.savetaper = savetaper
        if self.tapertype is not None:
            tap = taper2d(dimsd[1], nwin, nover, tapertype=self.tapertype)
            tapin = tap.copy()
            tapin[:nover] = 1
            tapend = tap.copy()
            tapend[-nover:] = 1
            if self.savetaper:
                self.taps = [
                    tapin[np.newaxis, :],
                ]
                for _ in range(1, nwins - 1):
                    self.taps.append(tap[np.newaxis, :])
                self.taps.append(tapend[np.newaxis, :])
                self.taps = np.concatenate(self.taps, axis=0)
            else:
                self.taps = np.vstack(
                    [tapin[np.newaxis, :], tap[np.newaxis, :], tapend[np.newaxis, :]]
                )

        # check if operator is applied to all windows simultaneously
        self.simOp = False
        if Op.shape[1] == np.prod(dims):
            self.simOp = True
        self.Op = Op

        super().__init__(
            dtype=Op.dtype,
            dims=(nwins, int(dims[0] // nwins), dims[1]),
            dimsd=dimsd,
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
            x = self.Op @ x
        for iwin0 in range(self.dims[0]):
            if self.simOp:
                xx = x[iwin0].reshape(self.nwin, self.dimsd[-1])
            else:
                xx = self.Op.matvec(x[iwin0].ravel()).reshape(self.nwin, self.dimsd[-1])
            if self.tapertype is not None:
                xxwin = self.taps[iwin0] * xx
            else:
                xxwin = xx
            y[self.dwin_inends[0][iwin0] : self.dwin_inends[1][iwin0]] += xxwin
        return y

    @reshaped
    def _rmatvec_savetaper(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        ncp_sliding_window_view = get_sliding_window_view(x)
        if self.tapertype is not None:
            self.taps = to_cupy_conditional(x, self.taps)
        ywins = ncp_sliding_window_view(x, self.nwin, axis=0)[
            :: self.nwin - self.nover
        ].transpose(0, 2, 1)
        if self.tapertype is not None:
            ywins = ywins * self.taps
        if self.simOp:
            y = self.Op.H @ ywins
        else:
            y = ncp.zeros(self.dims, dtype=self.dtype)
            for iwin0 in range(self.dims[0]):
                y[iwin0] = self.Op.rmatvec(ywins[iwin0].ravel()).reshape(
                    self.dims[1], self.dims[2]
                )
        return y

    @reshaped
    def _matvec_nosavetaper(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if self.tapertype is not None:
            self.taps = to_cupy_conditional(x, self.taps)
        y = ncp.zeros(self.dimsd, dtype=self.dtype)
        if self.simOp:
            x = self.Op @ x
        for iwin0 in range(self.dims[0]):
            if self.simOp:
                xxwin = x[iwin0].reshape(self.nwin, self.dimsd[-1])
            else:
                xxwin = self.Op.matvec(x[iwin0].ravel()).reshape(
                    self.nwin, self.dimsd[-1]
                )
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
        ywins = (
            ncp_sliding_window_view(x, self.nwin, axis=0)[:: self.nwin - self.nover]
            .transpose(0, 2, 1)
            .copy()
        )
        if self.simOp:
            if self.tapertype is not None:
                for iwin0 in range(self.dims[0]):
                    ywins = self._apply_taper(ywins, iwin0)
            y = self.Op.H @ ywins
        else:
            y = ncp.zeros(self.dims, dtype=self.dtype)
            for iwin0 in range(self.dims[0]):
                if self.tapertype is not None:
                    ywins = self._apply_taper(ywins, iwin0)
                y[iwin0] = self.Op.rmatvec(ywins[iwin0].ravel()).reshape(
                    self.dims[1], self.dims[2]
                )
        return y

    def _register_multiplications(self, savetaper: bool) -> None:
        if savetaper:
            self._matvec = self._matvec_savetaper
            self._rmatvec = self._rmatvec_savetaper
        else:
            self._matvec = self._matvec_nosavetaper
            self._rmatvec = self._rmatvec_nosavetaper
