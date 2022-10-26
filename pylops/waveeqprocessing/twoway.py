__all__ = ["AcousticWave2D"]

from typing import Tuple

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike

devito_message = deps.devito_import("the twoway module")

if devito_message is None:
    from examples.seismic import AcquisitionGeometry, Model
    from examples.seismic.acoustic import AcousticWaveSolver


class AcousticWave2D(LinearOperator):
    """Devito Acoustic propagator.

    Parameters
    ----------
    shape : :obj:`tuple` or :obj:`numpy.ndarray`
        Model shape ``(nx, nz)``
    origin : :obj:`tuple` or :obj:`numpy.ndarray`
        Model origin ``(ox, oz)``
    spacing : :obj:`tuple` or  :obj:`numpy.ndarray`
        Model spacing ``(dx, dz)``
    vp : :obj:`numpy.ndarray`
        Velocity model in m/s
    src_x : :obj:`numpy.ndarray`
        Source x-coordinates in m
    src_z : :obj:`numpy.ndarray` or :obj:`float`
        Source z-coordinates in m
    rec_x : :obj:`numpy.ndarray`
        Receiver x-coordinates in m
    rec_z : :obj:`numpy.ndarray` or :obj:`float`
        Receiver z-coordinates in m
    t0 : :obj:`float`
        Initial time
    tn : :obj:`int`
        Number of time samples
    src_type : :obj:`str`
        Source type
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    nbl : :obj:`int`, optional
        Number ordering of samples in absorbing boundaries
    f0 : :obj:`float`, optional
        Source peak frequency (Hz)
    checkpointing : :obj:`bool`, optional
        Use checkpointing (``True``) or not (``False``). Note that
        using checkpointing is needed when dealing with large models
        but it will slow down computations
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    """

    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: int,
        src_type: str = "Ricker",
        space_order: int = 6,
        nbl: int = 20,
        f0: float = 20.0,
        checkpointing: bool = False,
        dtype: DTypeLike = "float32",
        name: str = "A",
    ) -> None:
        if devito_message is not None:
            raise NotImplementedError(devito_message)

        # create model
        self._create_model(shape, origin, spacing, vp, space_order, nbl)
        self._create_geometry(src_x, src_z, rec_x, rec_z, t0, tn, src_type, f0=f0)
        self.checkpointing = checkpointing

        super().__init__(
            dtype=np.dtype(dtype),
            dims=vp.shape,
            dimsd=(len(src_x), len(rec_x), self.geometry.nt),
            explicit=False,
            name=name,
        )

    @staticmethod
    def _crop_model(m: NDArray, nbl: int) -> NDArray:
        """Remove absorbing boundaries from model"""
        return m[nbl:-nbl, nbl:-nbl]

    def _create_model(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        space_order: int = 6,
        nbl: int = 20,
    ) -> None:
        """Create model

        Parameters
        ----------
        shape : :obj:`numpy.ndarray`
            Model shape ``(nx, nz)``
        origin : :obj:`numpy.ndarray`
            Model origin ``(ox, oz)``
        spacing : :obj:`numpy.ndarray`
            Model spacing ``(dx, dz)``
        vp : :obj:`numpy.ndarray`
            Velocity model in m/s
        space_order : :obj:`int`, optional
            Spatial ordering of FD stencil
        nbl : :obj:`int`, optional
            Number ordering of samples in absorbing boundaries

        """
        self.space_order = space_order
        self.model = Model(
            space_order=space_order,
            vp=vp * 1e-3,
            origin=origin,
            shape=shape,
            dtype=np.float32,
            spacing=spacing,
            nbl=nbl,
            bcs="damp",
        )

    def _create_geometry(
        self,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: int,
        src_type: str,
        f0: float = 20.0,
    ) -> None:
        """Create geometry and time axis

        Parameters
        ----------
        src_x : :obj:`numpy.ndarray`
            Source x-coordinates in m
        src_z : :obj:`numpy.ndarray` or :obj:`float`
            Source z-coordinates in m
        rec_x : :obj:`numpy.ndarray`
            Receiver x-coordinates in m
        rec_z : :obj:`numpy.ndarray` or :obj:`float`
            Receiver z-coordinates in m
        t0 : :obj:`float`
            Initial time
        tn : :obj:`int`
            Number of time samples
        src_type : :obj:`str`
            Source type
        f0 : :obj:`float`, optional
            Source peak frequency (Hz)

        """

        nsrc, nrec = len(src_x), len(rec_x)
        src_coordinates = np.empty((nsrc, 2))
        src_coordinates[:, 0] = src_x
        src_coordinates[:, 1] = src_z

        rec_coordinates = np.empty((nrec, 2))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, 1] = rec_z

        self.geometry = AcquisitionGeometry(
            self.model,
            rec_coordinates,
            src_coordinates,
            t0,
            tn,
            src_type=src_type,
            f0=None if f0 is None else f0 * 1e-3,
        )

    def _srcillumination_oneshot(self, isrc: int) -> Tuple[NDArray, NDArray]:
        """Source wavefield and illumination for one shot

        Parameters
        ----------
        isrc : :obj:`int`
            Index of source to model

        Returns
        -------
        u0 : :obj:`np.ndarray`
            Source wavefield
        src_ill : :obj:`np.ndarray`
            Source illumination

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[isrc, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )
        solver = AcousticWaveSolver(self.model, geometry, space_order=self.space_order)

        # source wavefield
        u0 = solver.forward(save=True)[1]
        # source illumination
        src_ill = self._crop_model((u0.data**2).sum(axis=0), self.model.nbl)
        return u0, src_ill

    def srcillumination_allshots(self, savewav: bool = False) -> None:
        """Source wavefield and illumination for all shots

        Parameters
        ----------
        savewav : :obj:`bool`, optional
            Save source wavefield (``True``) or not (``False``)

        """
        nsrc = self.geometry.src_positions.shape[0]
        if savewav:
            self.src_wavefield = []
        self.src_illumination = np.zeros(self.model.shape)

        for isrc in range(nsrc):
            src_wav, src_ill = self._srcillumination_oneshot(isrc)
            if savewav:
                self.src_wavefield.append(src_wav)
            self.src_illumination += src_ill

    def _born_oneshot(self, isrc: int, dm: NDArray) -> NDArray:
        """Born modelling for one shot

        Parameters
        ----------
        isrc : :obj:`int`
            Index of source to model
        dm : :obj:`np.ndarray`
            Model perturbation

        Returns
        -------
        d : :obj:`np.ndarray`
            Data

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[isrc, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )

        # set perturbation
        dmext = np.zeros(self.model.grid.shape, dtype=np.float32)
        dmext[self.model.nbl : -self.model.nbl, self.model.nbl : -self.model.nbl] = dm

        # solve
        solver = AcousticWaveSolver(self.model, geometry, space_order=self.space_order)
        d = solver.jacobian(dmext)[0]
        d = d.resample(geometry.dt).data[:][: geometry.nt].T
        return d

    def _born_allshots(self, dm: NDArray) -> NDArray:
        """Born modelling for all shots

        Parameters
        -----------
        dm : :obj:`np.ndarray`
            Model perturbation

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data for all shots

        """
        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in range(nsrc):
            d = self._born_oneshot(isrc, dm)
            dtot.append(d)
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        return dtot

    def _bornadj_oneshot(self, isrc, dobs):
        """Adjoint born modelling for one shot

        Parameters
        ----------
        isrc : :obj:`float`
            Index of source to model
        dobs : :obj:`np.ndarray`
            Observed data to inject

        Returns
        -------
        model : :obj:`np.ndarray`
            Model

        """
        # create geometry for single source
        geometry = AcquisitionGeometry(
            self.model,
            self.geometry.rec_positions,
            self.geometry.src_positions[isrc, :],
            self.geometry.t0,
            self.geometry.tn,
            f0=self.geometry.f0,
            src_type=self.geometry.src_type,
        )
        # create boundary data
        recs = self.geometry.rec.copy()
        recs.data[:] = dobs.T[:]

        solver = AcousticWaveSolver(self.model, geometry, space_order=self.space_order)

        # source wavefield
        if hasattr(self, "src_wavefield"):
            u0 = self.src_wavefield[isrc]
        else:
            u0 = solver.forward(save=True)[1]
        # adjoint modelling (reverse wavefield plus imaging condition)
        model = solver.jacobian_adjoint(
            rec=recs, u=u0, checkpointing=self.checkpointing
        )[0]
        return model

    def _bornadj_allshots(self, dobs: NDArray) -> NDArray:
        """Adjoint Born modelling for all shots

        Parameters
        ----------
        dobs : :obj:`np.ndarray`
            Observed data to inject

        Returns
        -------
        model : :obj:`np.ndarray`
            Model

        """
        nsrc = self.geometry.src_positions.shape[0]
        mtot = np.zeros(self.model.shape, dtype=np.float32)

        for isrc in range(nsrc):
            m = self._bornadj_oneshot(isrc, dobs[isrc])
            mtot += self._crop_model(m.data, self.model.nbl)
        return mtot

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = self._born_allshots(x)
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        y = self._bornadj_allshots(x)
        return y
