__all__ = [
    "patch2d_design",
    "Patch2D",
]

import logging
from typing import Optional, Sequence, Tuple

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
from pylops.utils.tapers import taper2d
from pylops.utils.typing import InputDimsLike, NDArray

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def patch2d_design(
    dimsd: InputDimsLike,
    nwin: Tuple[int, int],
    nover: Tuple[int, int],
    nop: Tuple[int, int],
    verb: bool = True,
) -> Tuple[
    Tuple[int, int],
    Tuple[int, int],
    Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]],
    Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]],
]:
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
    verb : :obj:`bool`, optional
        Verbosity flag. If ``verb==True``, print the data
        and model windows start-end indices

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
    if verb:
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


class Patch2D(LinearOperator):
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
    savetaper : :obj:`bool`, optional
        .. versionadded:: 2.3.0

        Save all tapers and apply them in one go (``True``) or save unique tapers and apply them one by one (``False``).
        The first option is more computationally efficient, whilst the second is more memory efficient.
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

    def __init__(
        self,
        Op: LinearOperator,
        dims: InputDimsLike,
        dimsd: InputDimsLike,
        nwin: Tuple[int, int],
        nover: Tuple[int, int],
        nop: Tuple[int, int],
        tapertype: str = "hanning",
        savetaper: bool = True,
        scalings: Optional[Sequence[float]] = None,
        name: str = "P",
    ) -> None:

        dims: Tuple[int, ...] = _value_or_sized_to_tuple(dims)
        dimsd: Tuple[int, ...] = _value_or_sized_to_tuple(dimsd)

        # data windows
        dwin0_ins, dwin0_ends = _slidingsteps(dimsd[0], nwin[0], nover[0])
        dwin1_ins, dwin1_ends = _slidingsteps(dimsd[1], nwin[1], nover[1])
        self.dwins_inends = ((dwin0_ins, dwin0_ends), (dwin1_ins, dwin1_ends))
        nwins0 = len(dwin0_ins)
        nwins1 = len(dwin1_ins)
        nwins = nwins0 * nwins1
        self.nwin = nwin
        self.nover = nover

        # check patching
        if nwins0 * nop[0] != dims[0] or nwins1 * nop[1] != dims[1]:
            raise ValueError(
                f"Model shape (dims={dims}) is not consistent with chosen "
                f"number of windows. Run patch2d_design to identify the "
                f"correct number of windows for the current "
                "model size..."
            )

        # create tapers
        self.tapertype = tapertype
        self.savetaper = savetaper
        if self.tapertype is not None:
            tap = taper2d(nwin[1], nwin[0], nover, tapertype=tapertype).astype(Op.dtype)
            # topmost tapers
            taptop = tap.copy()
            taptop[: nover[0]] = tap[nwin[0] // 2]
            # bottommost tapers
            tapbottom = tap.copy()
            tapbottom[-nover[0] :] = tap[nwin[0] // 2]
            # leftmost tapers
            tapleft = tap.copy()
            tapleft[:, : nover[1]] = tap[:, nwin[1] // 2][:, np.newaxis]
            # rightmost tapers
            tapright = tap.copy()
            tapright[:, -nover[1] :] = tap[:, nwin[1] // 2][:, np.newaxis]
            # lefttopcorner taper
            taplefttop = tap.copy()
            taplefttop[:, : nover[1]] = tap[:, nwin[1] // 2][:, np.newaxis]
            taplefttop[: nover[0]] = taplefttop[nwin[0] // 2]
            # righttopcorner taper
            taprighttop = tap.copy()
            taprighttop[:, -nover[1] :] = tap[:, nwin[1] // 2][:, np.newaxis]
            taprighttop[: nover[0]] = taprighttop[nwin[0] // 2]
            # leftbottomcorner taper
            tapleftbottom = tap.copy()
            tapleftbottom[:, : nover[1]] = tap[:, nwin[1] // 2][:, np.newaxis]
            tapleftbottom[-nover[0] :] = tapleftbottom[nwin[0] // 2]
            # rightbottomcorner taper
            taprightbottom = tap.copy()
            taprightbottom[:, -nover[1] :] = tap[:, nwin[1] // 2][:, np.newaxis]
            taprightbottom[-nover[0] :] = taprightbottom[nwin[0] // 2]

            if self.savetaper:
                taps = [
                    tap,
                ] * nwins
                for itap in range(0, nwins1):
                    taps[itap] = taptop
                for itap in range(nwins - nwins1, nwins):
                    taps[itap] = tapbottom
                for itap in range(0, nwins, nwins1):
                    taps[itap] = tapleft
                for itap in range(nwins1 - 1, nwins, nwins1):
                    taps[itap] = tapright
                taps[0] = taplefttop
                taps[nwins1 - 1] = taprighttop
                taps[nwins - nwins1] = tapleftbottom
                taps[nwins - 1] = taprightbottom
                self.taps = np.vstack(taps).reshape(nwins0, nwins1, nwin[0], nwin[1])
            else:
                taps = [
                    taplefttop,
                    taptop,
                    taprighttop,
                    tapleft,
                    tap,
                    tapright,
                    tapleftbottom,
                    tapbottom,
                    taprightbottom,
                ]
                self.taps = np.vstack(taps).reshape(3, 3, nwin[0], nwin[1])

        # define scalings
        self.scalings = [1.0] * nwins if scalings is None else scalings

        # check if operator is applied to all windows simultaneously
        self.simOp = False
        if Op.shape[1] == np.prod(dims):
            self.simOp = True
        self.Op = Op

        super().__init__(
            dtype=Op.dtype,
            dims=(nwins0, nwins1, int(dims[0] // nwins0), int(dims[1] // nwins1)),
            dimsd=dimsd,
            clinear=False,
            name=name,
        )

        self._register_multiplications(self.savetaper)

    def _apply_taper(self, ywins, iwin0, iwin1):
        if iwin0 == 0 and iwin1 == 0:
            ywins[0, 0] = self.taps[0, 0] * ywins[0, 0]
        elif iwin0 == 0 and iwin1 == self.dims[1] - 1:
            ywins[0, -1] = self.taps[0, -1] * ywins[0, -1]
        elif iwin0 == 0:
            ywins[0, iwin1] = self.taps[0, 1] * ywins[0, iwin1]
        elif iwin0 == self.dims[0] - 1 and iwin1 == 0:
            ywins[-1, 0] = self.taps[-1, 0] * ywins[-1, 0]
        elif iwin0 == self.dims[0] - 1 and iwin1 == self.dims[1] - 1:
            ywins[-1, -1] = self.taps[-1, -1] * ywins[-1, -1]
        elif iwin0 == self.dims[0] - 1:
            ywins[-1, iwin1] = self.taps[-1, 1] * ywins[-1, iwin1]
        elif iwin1 == 0:
            ywins[iwin0, 0] = self.taps[1, 0] * ywins[iwin0, 0]
        elif iwin1 == self.dims[1] - 1:
            ywins[iwin0, -1] = self.taps[1, -1] * ywins[iwin0, -1]
        else:
            ywins[iwin0, iwin1] = self.taps[1, 1] * ywins[iwin0, iwin1]
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
            for iwin1 in range(self.dims[1]):
                if self.simOp:
                    xx = x[iwin0, iwin1].reshape(self.nwin)
                else:
                    xx = self.Op.matvec(x[iwin0, iwin1].ravel()).reshape(self.nwin)
                if self.tapertype is not None:
                    xxwin = self.taps[iwin0, iwin1] * xx
                else:
                    xxwin = xx

                y[
                    self.dwins_inends[0][0][iwin0] : self.dwins_inends[0][1][iwin0],
                    self.dwins_inends[1][0][iwin1] : self.dwins_inends[1][1][iwin1],
                ] += xxwin
        return y

    @reshaped
    def _rmatvec_savetaper(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        ncp_sliding_window_view = get_sliding_window_view(x)
        if self.tapertype is not None:
            self.taps = to_cupy_conditional(x, self.taps)
        ywins = ncp_sliding_window_view(x, self.nwin)[
            :: self.nwin[0] - self.nover[0], :: self.nwin[1] - self.nover[1]
        ]
        if self.tapertype is not None:
            ywins = ywins * self.taps
        if self.simOp:
            y = self.Op.H @ ywins
        else:
            y = ncp.zeros(self.dims, dtype=self.dtype)
            for iwin0 in range(self.dims[0]):
                for iwin1 in range(self.dims[1]):
                    y[iwin0, iwin1] = self.Op.rmatvec(
                        ywins[iwin0, iwin1].ravel()
                    ).reshape(self.dims[2], self.dims[3])
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
            for iwin1 in range(self.dims[1]):
                if self.simOp:
                    xxwin = x[iwin0, iwin1].reshape(self.nwin)
                else:
                    xxwin = self.Op.matvec(x[iwin0, iwin1].ravel()).reshape(self.nwin)
                if self.tapertype is not None:
                    if iwin0 == 0 and iwin1 == 0:
                        xxwin = self.taps[0, 0] * xxwin
                    elif iwin0 == 0 and iwin1 == self.dims[1] - 1:
                        xxwin = self.taps[0, -1] * xxwin
                    elif iwin0 == 0:
                        xxwin = self.taps[0, 1] * xxwin
                    elif iwin0 == self.dims[0] - 1 and iwin1 == 0:
                        xxwin = self.taps[-1, 0] * xxwin
                    elif iwin0 == self.dims[0] - 1 and iwin1 == self.dims[1] - 1:
                        xxwin = self.taps[-1, -1] * xxwin
                    elif iwin0 == self.dims[0] - 1:
                        xxwin = self.taps[-1, 1] * xxwin
                    elif iwin1 == 0:
                        xxwin = self.taps[1, 0] * xxwin
                    elif iwin1 == self.dims[1] - 1:
                        xxwin = self.taps[1, -1] * xxwin
                    else:
                        xxwin = self.taps[1, 1] * xxwin

                y[
                    self.dwins_inends[0][0][iwin0] : self.dwins_inends[0][1][iwin0],
                    self.dwins_inends[1][0][iwin1] : self.dwins_inends[1][1][iwin1],
                ] += xxwin
        return y

    @reshaped
    def _rmatvec_nosavetaper(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        ncp_sliding_window_view = get_sliding_window_view(x)
        if self.tapertype is not None:
            self.taps = to_cupy_conditional(x, self.taps)
        ywins = ncp_sliding_window_view(x, self.nwin)[
            :: self.nwin[0] - self.nover[0], :: self.nwin[1] - self.nover[1]
        ].copy()
        if self.simOp:
            if self.tapertype is not None:
                for iwin0 in range(self.dims[0]):
                    for iwin1 in range(self.dims[1]):
                        ywins = self._apply_taper(ywins, iwin0, iwin1)
            y = self.Op.H @ ywins
        else:
            y = ncp.zeros(self.dims, dtype=self.dtype)
            for iwin0 in range(self.dims[0]):
                for iwin1 in range(self.dims[1]):
                    if self.tapertype is not None:
                        ywins = self._apply_taper(ywins, iwin0, iwin1)
                    y[iwin0, iwin1] = self.Op.rmatvec(
                        ywins[iwin0, iwin1].ravel()
                    ).reshape(self.dims[2], self.dims[3])
        return y

    def _register_multiplications(self, savetaper: bool) -> None:
        if savetaper:
            self._matvec = self._matvec_savetaper
            self._rmatvec = self._rmatvec_savetaper
        else:
            self._matvec = self._matvec_nosavetaper
            self._rmatvec = self._rmatvec_nosavetaper
