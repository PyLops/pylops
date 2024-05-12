__all__ = [
    "sliding3d_design",
    "Sliding3D",
]

import logging
from typing import Tuple

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
from pylops.utils.tapers import taper3d
from pylops.utils.typing import InputDimsLike, NDArray

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def sliding3d_design(
    dimsd: Tuple[int, int, int],
    nwin: Tuple[int, int],
    nover: Tuple[int, int],
    nop: Tuple[int, int, int],
) -> Tuple[
    Tuple[int, int],
    Tuple[int, int, int],
    Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]],
    Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]],
]:
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


class Sliding3D(LinearOperator):
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

    .. note:: Two kind of operators ``Op`` can be provided: the first
       applies a single transformation to each window separately; the second
       applies the transformation to all of the windows at the same time. This
       is directly inferred during initialization when the following condition
       holds ``Op.shape[1] == np.prod(dims)``.

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
        *Deprecated*, will be removed in v3.0.0. Simply kept for
        back-compatibility with previous implementation
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
        nwin: Tuple[int, int],
        nover: Tuple[int, int],
        nop: Tuple[int, int, int],
        tapertype: str = "hanning",
        nproc: int = 1,
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

        # model windows
        mwin0_ins, mwin0_ends = _slidingsteps(dims[0], nop[0], 0)
        mwin1_ins, mwin1_ends = _slidingsteps(dims[1], nop[1], 0)
        self.mwins_inends = ((mwin0_ins, mwin0_ends), (mwin1_ins, mwin1_ends))

        # check windows
        if nwins * Op.shape[1] // dims[2] != dims[0] * dims[1] and Op.shape[
            1
        ] != np.prod(dims):
            raise ValueError(
                f"Model shape (dims={dims}) is not consistent with chosen "
                f"number of windows. Run sliding3d_design to identify the "
                f"correct number of windows for the current "
                "model size..."
            )

        # create tapers
        self.tapertype = tapertype
        if self.tapertype is not None:
            tap = taper3d(dimsd[2], nwin, nover, tapertype=tapertype).astype(Op.dtype)
            taps = [
                tap,
            ] * nwins

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
            self.taps = np.vstack(taps).reshape(
                nwins0, nwins1, nwin[0], nwin[1], dimsd[2]
            )

        # check if operator is applied to all windows simultaneously
        self.simOp = False
        if Op.shape[1] == np.prod(dims):
            self.simOp = True
        self.Op = Op

        super().__init__(
            dtype=Op.dtype,
            dims=(
                nwins0,
                nwins1,
                int(dims[0] // nwins0),
                int(dims[1] // nwins1),
                dims[2],
            ),
            dimsd=dimsd,
            clinear=False,
            name=name,
        )

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        if self.tapertype is not None:
            self.taps = to_cupy_conditional(x, self.taps)
        y = ncp.zeros(self.dimsd, dtype=self.dtype)
        if self.simOp:
            x = self.Op @ x
        for iwin0 in range(self.dims[0]):
            for iwin1 in range(self.dims[1]):
                if self.simOp:
                    xx = x[iwin0, iwin1].reshape(
                        self.nwin[0], self.nwin[1], self.dimsd[-1]
                    )
                else:
                    xx = self.Op.matvec(x[iwin0, iwin1].ravel()).reshape(
                        self.nwin[0], self.nwin[1], self.dimsd[-1]
                    )
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
    def _rmatvec(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        ncp_sliding_window_view = get_sliding_window_view(x)
        if self.tapertype is not None:
            self.taps = to_cupy_conditional(x, self.taps)
        ywins = ncp_sliding_window_view(x, self.nwin, axis=(0, 1))[
            :: self.nwin[0] - self.nover[0], :: self.nwin[1] - self.nover[1]
        ].transpose(0, 1, 3, 4, 2)
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
                    ).reshape(self.dims[2], self.dims[3], self.dims[4])
        return y
