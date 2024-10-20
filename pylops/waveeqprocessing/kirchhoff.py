__all__ = ["Kirchhoff"]


import logging
import os
import warnings
from typing import Optional, Tuple, Union

import numpy as np

from pylops import LinearOperator
from pylops.signalprocessing import Convolve1D
from pylops.utils import deps
from pylops.utils._internal import _value_or_sized_to_array
from pylops.utils.decorators import reshaped
from pylops.utils.tapers import taper
from pylops.utils.typing import DTypeLike, NDArray

skfmm_message = deps.skfmm_import("the kirchhoff module")
jit_message = deps.numba_import("the kirchhoff module")

if skfmm_message is None:
    import skfmm

if jit_message is None:
    from numba import jit, prange

    from ._kirchhoff_cuda import _kirchhoffCudaHelper
    # detect whether to use parallel or not
    numba_threads = int(os.getenv("NUMBA_NUM_THREADS", "1"))
    parallel = True if numba_threads != 1 else False
else:
    prange = range

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class Kirchhoff(LinearOperator):
    r"""Kirchhoff demigration operator.

    Kirchhoff-based demigration/migration operator. Uses a high-frequency
    approximation of the Green's function propagators based on traveltimes
    and amplitudes that are either computed internally by solving the Eikonal equation,
    or passed directly by the user (which can use any other propagation engine of choice).

    Parameters
    ----------
    z : :obj:`numpy.ndarray`
        Depth axis
    x : :obj:`numpy.ndarray`
        Spatial axis
    t : :obj:`numpy.ndarray`
        Time axis for data
    srcs : :obj:`numpy.ndarray`
        Sources in array of size :math:`\lbrack 2 (3) \times n_s \rbrack`
        The first axis should be ordered as (``y``,) ``x``, ``z``.
    recs : :obj:`numpy.ndarray`
        Receivers in array of size :math:`\lbrack 2 (3) \times n_r \rbrack`
        The first axis should be ordered as (``y``,) ``x``, ``z``.
    vel : :obj:`numpy.ndarray` or :obj:`float`
        Velocity model of size :math:`\lbrack (n_y\,\times)\; n_x
        \times n_z \rbrack` (or constant)
    wav : :obj:`numpy.ndarray`
        Wavelet.
    wavcenter : :obj:`int`
        Index of wavelet center
    y : :obj:`numpy.ndarray`
        Additional spatial axis (for 3-dimensional problems)
    mode : :obj:`str`, optional
        Computation mode (``analytic``, ``eikonal`` or ``byot``, see Notes for
        more details)
    wavfilter : :obj:`bool`, optional
        .. versionadded:: 2.0.0

        Apply wavelet filter (``True``) or not (``False``)
    dynamic : :obj:`bool`, optional
        .. versionadded:: 2.0.0

        Apply dynamic weights in computations (``True``) or not (``False``). This includes both the amplitude
        terms of the Green's function and the reflectivity-related scaling term (see equations below).
    trav : :obj:`numpy.ndarray` or :obj:`tuple`, optional
        Traveltime table of size
        :math:`\lbrack (n_y) n_x n_z \times n_s n_r \rbrack` or pair of traveltime tables
        of size :math:`\lbrack (n_y) n_x n_z \times n_s \rbrack` and :math:`\lbrack (n_y) n_x n_z \times n_r \rbrack`
        (to be provided if ``mode='byot'``). Note that the latter approach is recommended as less memory demanding
        than the former. Moreover, only ``mode='dynamic'`` is only possible when traveltimes are provided in
        the latter form.
    amp : :obj:`numpy.ndarray`, optional
        .. versionadded:: 2.0.0

        Pair of amplitude tables of size :math:`\lbrack (n_y) n_x n_z \times n_s \rbrack` and
        :math:`\lbrack (n_y) n_x n_z \times n_r \rbrack` (to be provided if ``mode='byot'``). Note that this parameter
        is only required when ``mode='dynamic'`` is chosen.
    aperture : :obj:`float` or :obj:`tuple`, optional
        .. versionadded:: 2.0.0

        Maximum allowed aperture expressed as the ratio of offset over depth. If ``None``,
        no aperture limitations are introduced. If scalar, a taper from 80% to 100% of
        aperture is applied. If tuple, apertures below the first value are
        accepted and those after the second value are rejected. A tapering is implemented
        for those between such values.
    angleaperture : :obj:`float` or :obj:`tuple`, optional
        .. versionadded:: 2.0.0

        Maximum allowed angle (either source or receiver side) in degrees. If ``None``,
        angle aperture limitations are not introduced. See ``aperture`` for implementation
        details regarding scalar and tuple cases.
    snell : :obj:`float` or :obj:`tuple`, optional
        Deprecated, will be removed in v3.0.0. Simply kept for back-compatibility with previous implementation,
        but effectively not affecting the behaviour of the operator.
    engine : :obj:`str`, optional
        Engine used for computations (``numpy`` or ``numba``).
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Raises
    ------
    NotImplementedError
        If ``mode`` is neither ``analytic``, ``eikonal``, or ``byot``

    Notes
    -----
    The Kirchhoff demigration operator synthesizes seismic data given a
    propagation velocity model :math:`v(\mathbf{x})` and a
    reflectivity model :math:`m(\mathbf{x})`. In forward mode [1]_, [2]_, [3]_:

    .. math::
        d(\mathbf{x_r}, \mathbf{x_s}, t) =
        \widetilde{w}(t) * \int_V \frac{2 \cos\theta} {v(\mathbf{x})}
        G(\mathbf{x_r}, \mathbf{x}, t) G(\mathbf{x}, \mathbf{x_s}, t)
        m(\mathbf{x})  \,\mathrm{d}\mathbf{x}

    where :math:`G(\mathbf{x}, \mathbf{x_s}, t)` and :math:`G(\mathbf{x_r}, \mathbf{x}, t)`
    are the Green's functions from source-to-subsurface-to-receiver and finally
    :math:`\widetilde{w}(t)` is either a filtered version of the wavelet :math:`w(t)`
    as explained below (``wavfilter=True``) or the wavelet itself when (``wavfilter=False``).
    Moreover, an angle scaling is included in the modelling operator,
    where the reflection angle :math:`\theta=(\theta_s-\theta_r)/2` is half of the opening angle,
    with :math:`\theta_s` and :math:`\theta_r` representing the angles between the source-side
    and receiver-side rays and the vertical at the image point, respectively.

    In our implementation, the following high-frequency approximation of the
    Green's functions is adopted:

    .. math::
        G(\mathbf{x_r}, \mathbf{x}, \omega) = a(\mathbf{x_r}, \mathbf{x})
            e^{j \omega t(\mathbf{x_r}, \mathbf{x})}

    where :math:`a(\mathbf{x_r}, \mathbf{x})` is the amplitude and
    :math:`t(\mathbf{x_r}, \mathbf{x})` is the traveltime. When ``dynamic=False``, the
    amplitude correction terms are disregarded leading to a kinematic-only Kirchhoff operator.

    .. math::
        d(\mathbf{x_r}, \mathbf{x_s}, t) =
        \tilde{w}(t) * \int_V e^{j \omega (t(\mathbf{x_r}, \mathbf{x}) +
        t(\mathbf{x}, \mathbf{x_s}))} m(\mathbf{x}) \,\mathrm{d}\mathbf{x}

    On the  other hand, when ``dynamic=True``, the amplitude scaling is defined as

    * ``2D``: :math:`a(\mathbf{x}, \mathbf{y})=\frac{1}{\sqrt{\text{dist}(\mathbf{x}, \mathbf{y})}}`
    * ``3D``: :math:`a(\mathbf{x}, \mathbf{y})=\frac{1}{\text{dist}(\mathbf{x}, \mathbf{y})}`

    approximating the geometrical spreading of the wavefront. For ``mode=analytic``,
    :math:`\text{dist}(\mathbf{x}, \mathbf{y})=\|\mathbf{x} - \mathbf{y}\|`, whilst for
    ``mode=eikonal``, this is computed internally by the Eikonal solver.

    The wavelet filtering is applied as follows [4]_:

    * ``2D``: :math:`\tilde{W}(f)=\sqrt{j\omega} \cdot W(f)`
    * ``3D``: :math:`\tilde{W}(f)=-j\omega \cdot W(f)`

    Depending on the choice of ``mode`` the traveltime and amplitude of the Green's
    function will be also computed differently:

    * ``mode=analytic`` or ``mode=eikonal``: traveltimes, amplitudes, and angles
      are computed for every source-image point-receiver triplets upfront and the
      Green's functions are implemented from traveltime and amplitude look-up tables,
      placing scaled reflectivity values at corresponding source-to-receiver time
      in the data.
    * ``mode=byot``: bring your own tables. Traveltime table are provided
      directly by user using ``trav`` input parameter. Similarly, in this case one
      can also provide their own amplitude scaling ``amp``.

    Two aperture limitations have been also implemented as defined by:

    * ``aperture``: the maximum allowed aperture is expressed as the ratio of
      offset over depth. This aperture limitation avoid including grazing angles
      whose contributions can introduce aliasing effects. A taper is added at the
      edges of the aperture;
    * ``angleaperture``: the maximum allowed angle aperture is expressed as the
      difference between the incident or emerging angle at every image point and
      the vertical axis. This aperture limitation also avoid including grazing angles
      whose contributions can introduce aliasing effects. Note that for a homogenous
      medium and slowly varying heterogeneous medium the offset and angle aperture limits
      may work in the same way.

    Finally, the adjoint of the demigration operator is a *migration* operator which
    projects data in the model domain creating an image of the subsurface
    reflectivity.

    .. [1] Bleistein, N., Cohen, J.K., and Stockwell, J.W.
       "Mathematics of Multidimensional Seismic Imaging, Migration and
       Inversion", 2000.

    .. [2] Santos, L.T., Schleicher, J., Tygel, M., and Hubral, P.
       "Seismic modeling by demigration", Geophysics, 65(4), pp. 1281-1289, 2000.

    .. [3] Yang, K., and Zhang, J. "Comparison between Born and Kirchhoff operators for
       least-squares reverse time migration and the constraint of the propagation of the
       background wavefield", Geophysics, 84(5), pp. R725-R739, 2019.

    .. [4] Safron, L. "Multicomponent least-squares Kirchhoff depth migration",
       MSc Thesis, 2018.

    """

    def __init__(
        self,
        z: NDArray,
        x: NDArray,
        t: NDArray,
        srcs: NDArray,
        recs: NDArray,
        vel: NDArray,
        wav: NDArray,
        wavcenter: int,
        y: Optional[NDArray] = None,
        mode: str = "eikonal",
        wavfilter: bool = False,
        dynamic: bool = False,
        trav: Optional[NDArray] = None,
        amp: Optional[NDArray] = None,
        aperture: Optional[Tuple[float, float]] = None,
        angleaperture: Union[float, Tuple[float, float]] = 90.0,
        snell: Optional[Tuple[float, float]] = None,
        engine: str = "numpy",
        dtype: DTypeLike = "float64",
        name: str = "K",
    ) -> None:
        warnings.warn(
            "A new implementation of Kirchhoff is provided in v2.1.0. "
            "This currently affects only the inner working of the "
            "operator, end-users can continue using the operator in "
            "the same way. Nevertheless, it is now recommended to provide"
            "the variables trav (and amp) as a tuples containing the "
            "traveltime (and amplitude) tables for sources and receivers "
            "separately. This behaviour will eventually become default in "
            "version v3.0.0.",
            FutureWarning,
        )
        # identify geometry
        (
            self.ndims,
            _,
            dims,
            self.ny,
            self.nx,
            self.nz,
            ns,
            nr,
            _,
            _,
            _,
            _,
            _,
        ) = Kirchhoff._identify_geometry(z, x, srcs, recs, y=y)
        self.dt = t[1] - t[0]
        self.nt = len(t)

        # store ix-iy locations of sources and receivers for aperture filter
        self.dynamic = dynamic
        if self.dynamic:
            dx = x[1] - x[0]
            if self.ndims == 2:
                self.six = (
                    np.tile((srcs[0] - x[0]) // dx, (nr, 1)).T.astype(int).ravel()
                )
                self.rix = np.tile((recs[0] - x[0]) // dx, (ns, 1)).astype(int).ravel()
            elif self.ndims == 3:
                # TODO: compute 3D indices for aperture filter
                # currently no aperture filter in 3D... just make indices 0
                # so check if always passed
                self.six = np.zeros(nr * ns)
                self.rix = np.zeros(nr * ns)

        # compute traveltime and distances
        self.travsrcrec = True  # use separate tables for src and rec traveltimes
        if mode in ["analytic", "eikonal", "byot"]:
            if mode in ["analytic", "eikonal"]:
                # compute traveltime table
                (
                    self.trav_srcs,
                    self.trav_recs,
                    dist_srcs,
                    dist_recs,
                    trav_srcs_grad,
                    trav_recs_grad,
                ) = Kirchhoff._traveltime_table(z, x, srcs, recs, vel, y=y, mode=mode)
                if self.dynamic:
                    # need to add a scalar in the denominator of amplitude term to avoid
                    # division by 0, currently set to 1e-2 of max distance to avoid having
                    # too large scaling around the source. This number may change in future
                    # or left to the user to define
                    epsdist = 1e-2
                    self.maxdist = epsdist * (np.max(dist_srcs) + np.max(dist_recs))
                    if self.ndims == 2:
                        self.amp_srcs, self.amp_recs = 1.0 / np.sqrt(
                            dist_srcs + self.maxdist
                        ), 1.0 / np.sqrt(dist_recs + self.maxdist)
                    else:
                        self.amp_srcs, self.amp_recs = 1.0 / (
                            dist_srcs + self.maxdist
                        ), 1.0 / (dist_recs + self.maxdist)
            else:
                if isinstance(trav, tuple):
                    self.trav_srcs, self.trav_recs = trav
                else:
                    self.travsrcrec = False
                    self.trav = trav

                if self.dynamic and not self.travsrcrec:
                    raise NotImplementedError(
                        "separate traveltime tables must be provided "
                        "when selecting mode=dynamic"
                    )
                if self.dynamic:
                    if isinstance(amp, tuple):
                        self.amp_srcs, self.amp_recs = amp
                    else:
                        raise NotImplementedError(
                            "separate amplitude tables must be provided "
                        )

                    if self.travsrcrec:
                        # compute traveltime gradients at image points
                        trav_srcs_grad = np.gradient(
                            self.trav_srcs.reshape(*dims, ns),
                            axis=np.arange(self.ndims),
                        )
                        trav_recs_grad = np.gradient(
                            self.trav_recs.reshape(*dims, nr),
                            axis=np.arange(self.ndims),
                        )
        else:
            raise NotImplementedError("method must be analytic, eikonal or byot")

        # compute angles with vertical
        if self.dynamic:
            if self.ndims == 2:
                self.angle_srcs = np.arctan2(
                    trav_srcs_grad[0], trav_srcs_grad[1]
                ).reshape(np.prod(dims), ns)
                self.angle_recs = np.arctan2(
                    trav_recs_grad[0], trav_recs_grad[1]
                ).reshape(np.prod(dims), nr)
            else:
                trav_srcs_grad = np.concatenate(
                    [trav_srcs_grad[i][np.newaxis] for i in range(3)]
                )
                trav_recs_grad = np.concatenate(
                    [trav_recs_grad[i][np.newaxis] for i in range(3)]
                )
                self.angle_srcs = (
                    np.sign(trav_srcs_grad[1])
                    * np.arccos(
                        trav_srcs_grad[-1]
                        / np.sqrt(np.sum(trav_srcs_grad**2, axis=0))
                    )
                ).reshape(np.prod(dims), ns)
                self.angle_recs = (
                    np.sign(trav_srcs_grad[1])
                    * np.arccos(
                        trav_recs_grad[-1]
                        / np.sqrt(np.sum(trav_recs_grad**2, axis=0))
                    )
                ).reshape(np.prod(dims), nr)

        # pre-compute traveltime indices if total traveltime is used
        if not self.travsrcrec:
            self.itrav = (self.trav / self.dt).astype("int32")
            self.travd = self.trav / self.dt - self.itrav

        # create wavelet operator
        if wavfilter:
            self.wav = self._wavelet_reshaping(
                wav, self.dt, srcs.shape[0], recs.shape[0], self.ndims
            )
        else:
            self.wav = wav
        self.cop = Convolve1D(
            (ns * nr, self.nt), h=self.wav, offset=wavcenter, axis=1, dtype=dtype
        )

        # create fixed-size aperture taper for all apertures
        self.aperturetap = taper(41, 20, "hanning")[20:]

        # define aperture
        # if aperture=None, we want to ensure the check is always matched (no aperture limits...)
        # if aperture!=None in 3d, force to None as aperture checks are not yet implemented
        if aperture is not None and self.ndims == 3:
            aperture = None
            warnings.warn(
                "Aperture is forced to None as currently not implemented in 3D"
            )
        if aperture is not None:
            warnings.warn(
                "Aperture is currently defined as ratio of offset over depth, "
                "and may be not ideal for highly heterogeneous media"
            )
        self.aperture = (
            (self.nx + 1,) if aperture is None else _value_or_sized_to_array(aperture)
        )
        if len(self.aperture) == 1:
            self.aperture = np.array([0.8 * self.aperture[0], self.aperture[0]])

        # define angle aperture
        angleaperture = [0.0, 1000.0] if angleaperture is None else angleaperture
        self.angleaperture = np.deg2rad(_value_or_sized_to_array(angleaperture))
        if len(self.angleaperture) == 1:
            self.angleaperture = np.array(
                [0.8 * self.angleaperture[0], self.angleaperture[0]]
            )

        # dimensions
        self.ns, self.nr = ns, nr
        self.nsnr = ns * nr
        self.ni = np.prod(dims)
        dims = tuple(dims) if self.ndims == 2 else (dims[0] * dims[1], dims[2])
        dimsd = (ns, nr, self.nt)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)
        # save velocity if using dynamic to compute amplitudes
        if self.dynamic:
            self.vel = (
                vel.flatten()
                if not isinstance(vel, (float, int))
                else vel * np.ones(np.prod(dims))
            )
        self._register_multiplications(engine)

    @staticmethod
    def _identify_geometry(
        z: NDArray,
        x: NDArray,
        srcs: NDArray,
        recs: NDArray,
        y: Optional[NDArray] = None,
    ) -> Tuple[
        int,
        int,
        NDArray,
        int,
        int,
        int,
        int,
        int,
        float,
        float,
        float,
        NDArray,
        NDArray,
    ]:
        """Identify geometry and acquisition size and sampling"""
        ns, nr = srcs.shape[1], recs.shape[1]
        nz, nx = len(z), len(x)
        dz = np.abs(z[1] - z[0])
        dx = np.abs(x[1] - x[0])
        if y is None:
            ndims = 2
            shiftdim = 0
            ny = 1
            dy = None
            dims = np.array([nx, nz])
            dsamp = np.array([dx, dz])
            origin = np.array([x[0], z[0]])
        else:
            ndims = 3
            shiftdim = 1
            ny = len(y)
            dy = np.abs(y[1] - y[0])
            dims = np.array([ny, nx, nz])
            dsamp = np.array([dy, dx, dz])
            origin = np.array([y[0], x[0], z[0]])
        return ndims, shiftdim, dims, ny, nx, nz, ns, nr, dy, dx, dz, dsamp, origin

    @staticmethod
    def _traveltime_table(
        z: NDArray,
        x: NDArray,
        srcs: NDArray,
        recs: NDArray,
        vel: Union[float, NDArray],
        y: Optional[NDArray] = None,
        mode: str = "eikonal",
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        r"""Traveltime table

        Compute traveltimes along the source-subsurface-receivers triplet
        in 2- or 3-dimensional media given a constant or depth- and space variable
        velocity.

        Parameters
        ----------
        z : :obj:`numpy.ndarray`
            Depth axis
        x : :obj:`numpy.ndarray`
            Spatial axis
        srcs : :obj:`numpy.ndarray`
            Sources in array of size :math:`\lbrack 2 (3) \times n_s \rbrack`
        recs : :obj:`numpy.ndarray`
            Receivers in array of size :math:`\lbrack 2 (3) \times n_r \rbrack`
        vel : :obj:`numpy.ndarray` or :obj:`float`
            Velocity model of size :math:`\lbrack (n_y \times)\, n_x
            \times n_z \rbrack` (or constant)
        y : :obj:`numpy.ndarray`
            Additional spatial axis (for 3-dimensional problems)
        mode : :obj:`numpy.ndarray`, optional
            Computation mode (``eikonal``, ``analytic`` - only for constant velocity)

        Returns
        -------
        trav_srcs : :obj:`numpy.ndarray`
            Source-to-subsurface traveltime table of size
            :math:`\lbrack (n_y*) n_x n_z \times n_s \rbrack`
        trav_recs : :obj:`numpy.ndarray`
            Receiver-to-subsurface traveltime table of size
            :math:`\lbrack (n_y) n_x n_z \times n_r \rbrack`
        dist_srcs : :obj:`numpy.ndarray`
            Source-to-subsurface distance table of size
            :math:`\lbrack (n_y*) n_x n_z \times n_s \rbrack`
        dist_recs : :obj:`numpy.ndarray`
            Receiver-to-subsurface distance table of size
            :math:`\lbrack (n_y) n_x n_z \times n_r \rbrack`
        trav_srcs_gradient : :obj:`numpy.ndarray`
            Source-to-subsurface traveltime gradient table of size
            :math:`\lbrack (n_y*) n_x n_z \times n_s \rbrack`
        trav_recs_gradient : :obj:`numpy.ndarray`
            Receiver-to-subsurface traveltime gradient table of size
            :math:`\lbrack (n_y) n_x n_z \times n_r \rbrack`

        """
        # define geometry
        (
            ndims,
            shiftdim,
            dims,
            ny,
            nx,
            nz,
            ns,
            nr,
            _,
            _,
            _,
            dsamp,
            origin,
        ) = Kirchhoff._identify_geometry(z, x, srcs, recs, y=y)

        # compute traveltimes
        if mode == "analytic":
            if not isinstance(vel, (float, int)):
                raise ValueError("vel must be scalar for mode=analytical")

            # compute grid
            if ndims == 2:
                X, Z = np.meshgrid(x, z, indexing="ij")
                X, Z = X.ravel(), Z.ravel()
            else:
                Y, X, Z = np.meshgrid(y, x, z, indexing="ij")
                Y, X, Z = Y.ravel(), X.ravel(), Z.ravel()

            dist_srcs2 = np.zeros((ny * nx * nz, ns))
            dist_recs2 = np.zeros((ny * nx * nz, nr))
            for isrc, src in enumerate(srcs.T):
                dist_srcs2[:, isrc] = (X - src[0 + shiftdim]) ** 2 + (
                    Z - src[1 + shiftdim]
                ) ** 2
                if ndims == 3:
                    dist_srcs2[:, isrc] += (Y - src[0]) ** 2
            for irec, rec in enumerate(recs.T):
                dist_recs2[:, irec] = (X - rec[0 + shiftdim]) ** 2 + (
                    Z - rec[1 + shiftdim]
                ) ** 2
                if ndims == 3:
                    dist_recs2[:, irec] += (Y - rec[0]) ** 2

            trav_srcs = np.sqrt(dist_srcs2) / vel
            trav_recs = np.sqrt(dist_recs2) / vel

            dist_srcs = trav_srcs * vel
            dist_recs = trav_recs * vel

        elif mode == "eikonal":
            if skfmm_message is None:
                dist_srcs = np.zeros((ny * nx * nz, ns))
                dist_recs = np.zeros((ny * nx * nz, nr))
                trav_srcs = np.zeros((ny * nx * nz, ns))
                trav_recs = np.zeros((ny * nx * nz, nr))
                for isrc, src in enumerate(srcs.T):
                    src = np.round((src - origin) / dsamp).astype(np.int32)
                    phi = np.ones_like(vel)
                    if ndims == 2:
                        phi[src[0], src[1]] = -1
                    else:
                        phi[src[0], src[1], src[2]] = -1
                    dist_srcs[:, isrc] = (skfmm.distance(phi=phi, dx=dsamp)).ravel()
                    trav_srcs[:, isrc] = (
                        skfmm.travel_time(phi=phi, speed=vel, dx=dsamp)
                    ).ravel()
                for irec, rec in enumerate(recs.T):
                    rec = np.round((rec - origin) / dsamp).astype(np.int32)
                    phi = np.ones_like(vel)
                    if ndims == 2:
                        phi[rec[0], rec[1]] = -1
                    else:
                        phi[rec[0], rec[1], rec[2]] = -1
                    dist_recs[:, irec] = (skfmm.distance(phi=phi, dx=dsamp)).ravel()
                    trav_recs[:, irec] = (
                        skfmm.travel_time(phi=phi, speed=vel, dx=dsamp)
                    ).ravel()
            else:
                raise NotImplementedError(skfmm_message)
        else:
            raise NotImplementedError("method must be analytic or eikonal")

        # compute traveltime gradients at image points
        trav_srcs_grad = np.gradient(
            trav_srcs.reshape(*dims, ns), *dsamp, axis=np.arange(ndims)
        )
        trav_recs_grad = np.gradient(
            trav_recs.reshape(*dims, nr), *dsamp, axis=np.arange(ndims)
        )

        return (
            trav_srcs,
            trav_recs,
            dist_srcs,
            dist_recs,
            trav_srcs_grad,
            trav_recs_grad,
        )

    def _wavelet_reshaping(
        self,
        wav: NDArray,
        dt: float,
        dimsrc: int,
        dimrec: int,
        dimv: int,
    ) -> NDArray:
        """Apply wavelet reshaping to account for omega scaling factor
        originating from the wave equation"""
        f = np.fft.rfftfreq(len(wav), dt)
        W = np.fft.rfft(wav, n=len(wav))
        if dimsrc == 2 and dimv == 2:
            # 2D
            Wfilt = W * np.sqrt(1j * 2 * np.pi * f)
        elif (dimsrc == 2 or dimrec == 2) and dimv == 3:
            # 2.5D
            raise NotImplementedError("2.D wavelet currently not available")
        elif dimsrc == 3 and dimrec == 3 and dimv == 3:
            # 3D
            Wfilt = W * (-1j * 2 * np.pi * f)
        wavfilt = np.fft.irfft(Wfilt, n=len(wav))
        return wavfilt

    @staticmethod
    def _trav_kirch_matvec(
        x: NDArray,
        y: NDArray,
        nsnr: int,
        nt: int,
        ni: int,
        itrav: NDArray,
        travd: NDArray,
    ) -> NDArray:
        for isrcrec in prange(nsnr):
            itravisrcrec = itrav[:, isrcrec]
            travdisrcrec = travd[:, isrcrec]
            for ii in range(ni):
                itravii = itravisrcrec[ii]
                travdii = travdisrcrec[ii]
                if 0 <= itravii < nt - 1:
                    y[isrcrec, itravii] += x[ii] * (1 - travdii)
                    y[isrcrec, itravii + 1] += x[ii] * travdii
        return y

    @staticmethod
    def _trav_kirch_rmatvec(
        x: NDArray,
        y: NDArray,
        nsnr: int,
        nt: int,
        ni: int,
        itrav: NDArray,
        travd: NDArray,
    ) -> NDArray:
        for ii in prange(ni):
            itravii = itrav[ii]
            travdii = travd[ii]
            for isrcrec in range(nsnr):
                itravisrcrecii = itravii[isrcrec]
                travdisrcrecii = travdii[isrcrec]
                if 0 <= itravisrcrecii < nt - 1:
                    y[ii] += (
                        x[isrcrec, itravisrcrecii] * (1 - travdisrcrecii)
                        + x[isrcrec, itravisrcrecii + 1] * travdisrcrecii
                    )
        return y

    @staticmethod
    def _travsrcrec_kirch_matvec(
        x: NDArray,
        y: NDArray,
        ns: int,
        nr: int,
        nt: int,
        ni: int,
        dt: float,
        trav_srcs: NDArray,
        trav_recs: NDArray,
    ) -> NDArray:
        for isrc in prange(ns):
            travisrc = trav_srcs[:, isrc]
            for irec in range(nr):
                travirec = trav_recs[:, irec]
                trav = travisrc + travirec
                itrav = (trav / dt).astype("int32")
                travd = trav / dt - itrav
                for ii in range(ni):
                    itravii = itrav[ii]
                    travdii = travd[ii]
                    if 0 <= itravii < nt - 1:
                        y[isrc * nr + irec, itravii] += x[ii] * (1 - travdii)
                        y[isrc * nr + irec, itravii + 1] += x[ii] * travdii
        return y

    @staticmethod
    def _travsrcrec_kirch_rmatvec(
        x: NDArray,
        y: NDArray,
        ns: int,
        nr: int,
        nt: int,
        ni: int,
        dt: float,
        trav_srcs: NDArray,
        trav_recs: NDArray,
    ) -> NDArray:
        for ii in prange(ni):
            trav_srcsii = trav_srcs[ii]
            trav_recsii = trav_recs[ii]
            for isrc in prange(ns):
                trav_srcii = trav_srcsii[isrc]
                for irec in range(nr):
                    trav_recii = trav_recsii[irec]
                    travii = trav_srcii + trav_recii
                    itravii = int(travii / dt)
                    travdii = travii / dt - itravii
                    if 0 <= itravii < nt - 1:
                        y[ii] += (
                            x[isrc * nr + irec, itravii] * (1 - travdii)
                            + x[isrc * nr + irec, itravii + 1] * travdii
                        )
        return y

    @staticmethod
    def _ampsrcrec_kirch_matvec(
        x: NDArray,
        y: NDArray,
        ns: int,
        nr: int,
        nt: int,
        ni: int,
        dt: float,
        vel: NDArray,
        trav_srcs: NDArray,
        trav_recs: NDArray,
        amp_srcs: NDArray,
        amp_recs: NDArray,
        aperturemin: float,
        aperturemax: float,
        aperturetap: NDArray,
        nz: int,
        six: NDArray,
        rix: NDArray,
        angleaperturemin: float,
        angleaperturemax: float,
        angles_srcs: NDArray,
        angles_recs: NDArray,
    ) -> NDArray:
        daperture = aperturemax - aperturemin
        dangleaperture = angleaperturemax - angleaperturemin
        for isrc in prange(ns):
            travisrc = trav_srcs[:, isrc]
            ampisrc = amp_srcs[:, isrc]
            angleisrc = angles_srcs[:, isrc]
            for irec in range(nr):
                travirec = trav_recs[:, irec]
                trav = travisrc + travirec
                itrav = (trav / dt).astype("int32")
                travd = trav / dt - itrav
                ampirec = amp_recs[:, irec]
                angleirec = angles_recs[:, irec]
                sixisrcrec = six[isrc * nr + irec]
                rixisrcrec = rix[isrc * nr + irec]
                # compute cosine of half opening angle and total amplitude scaling
                cosangle = np.cos((angleisrc - angleirec) / 2.0)
                amp = 2.0 * cosangle * ampisrc * ampirec / vel
                for ii in range(ni):
                    itravii = itrav[ii]
                    travdii = travd[ii]
                    damp = amp[ii]
                    # extract source and receiver angle at given image point
                    angle_src = angleisrc[ii]
                    angle_rec = angleirec[ii]
                    abs_angle_src = abs(angle_src)
                    abs_angle_rec = abs(angle_rec)
                    # angle apertures checks
                    aptap = 1.0
                    if (
                        abs_angle_src < angleaperturemax
                        and abs_angle_rec < angleaperturemax
                    ):
                        if abs_angle_src >= angleaperturemin:
                            # extract source angle aperture taper value
                            aptap = (
                                aptap
                                * aperturetap[
                                    int(
                                        20
                                        * (abs_angle_src - angleaperturemin)
                                        // dangleaperture
                                    )
                                ]
                            )
                        if abs_angle_rec >= angleaperturemin:
                            # extract receiver angle aperture taper value
                            aptap = (
                                aptap
                                * aperturetap[
                                    int(
                                        20
                                        * (abs_angle_rec - angleaperturemin)
                                        // dangleaperture
                                    )
                                ]
                            )

                        # identify x-index of image point
                        iz = ii % nz
                        # aperture check
                        aperture = abs(sixisrcrec - rixisrcrec) / (iz + 1)
                        if aperture < aperturemax:
                            if aperture >= aperturemin:
                                # extract aperture taper value
                                aptap = (
                                    aptap
                                    * aperturetap[
                                        int(
                                            20 * ((aperture - aperturemin) // daperture)
                                        )
                                    ]
                                )
                            # time limit check
                            if 0 <= itravii < nt - 1:
                                y[isrc * nr + irec, itravii] += (
                                    x[ii] * (1 - travdii) * damp * aptap
                                )
                                y[isrc * nr + irec, itravii + 1] += (
                                    x[ii] * travdii * damp * aptap
                                )
        return y

    @staticmethod
    def _ampsrcrec_kirch_rmatvec(
        x: NDArray,
        y: NDArray,
        ns: int,
        nr: int,
        nt: int,
        ni: int,
        dt: float,
        vel: NDArray,
        trav_srcs: NDArray,
        trav_recs: NDArray,
        amp_srcs: NDArray,
        amp_recs: NDArray,
        aperturemin: float,
        aperturemax: float,
        aperturetap: NDArray,
        nz: int,
        six: NDArray,
        rix: NDArray,
        angleaperturemin: float,
        angleaperturemax: float,
        angles_srcs: NDArray,
        angles_recs: NDArray,
    ) -> NDArray:
        daperture = aperturemax - aperturemin
        dangleaperture = angleaperturemax - angleaperturemin
        for ii in prange(ni):
            trav_srcsii = trav_srcs[ii]
            trav_recsii = trav_recs[ii]
            amp_srcsii = amp_srcs[ii]
            amp_recsii = amp_recs[ii]
            velii = vel[ii]
            angle_srcsii = angles_srcs[ii]
            angle_recsii = angles_recs[ii]
            # identify x-index of image point
            iz = ii % nz
            for isrc in range(ns):
                trav_srcii = trav_srcsii[isrc]
                amp_srcii = amp_srcsii[isrc]
                angle_src = angle_srcsii[isrc]
                for irec in range(nr):
                    trav_recii = trav_recsii[irec]
                    travii = trav_srcii + trav_recii
                    itravii = int(travii / dt)
                    travdii = travii / dt - itravii
                    amp_recii = amp_recsii[irec]
                    angle_rec = angle_recsii[irec]
                    sixisrcrec = six[isrc * nr + irec]
                    rixisrcrec = rix[isrc * nr + irec]
                    abs_angle_src = abs(angle_src)
                    abs_angle_rec = abs(angle_rec)
                    # compute cosine of half opening angle and total amplitude scaling
                    cosangle = np.cos((angle_src - angle_rec) / 2.0)
                    damp = 2.0 * cosangle * amp_srcii * amp_recii / velii
                    # angle apertures checks
                    aptap = 1.0
                    if (
                        abs_angle_src < angleaperturemax
                        and abs_angle_rec < angleaperturemax
                    ):
                        if abs_angle_src >= angleaperturemin:
                            # extract source angle aperture taper value
                            aptap = (
                                aptap
                                * aperturetap[
                                    int(
                                        20
                                        * (abs_angle_src - angleaperturemin)
                                        // dangleaperture
                                    )
                                ]
                            )
                        if abs_angle_rec >= angleaperturemin:
                            # extract receiver angle aperture taper value
                            aptap = (
                                aptap
                                * aperturetap[
                                    int(
                                        20
                                        * (abs_angle_rec - angleaperturemin)
                                        // dangleaperture
                                    )
                                ]
                            )

                        # aperture check
                        aperture = abs(sixisrcrec - rixisrcrec) / (iz + 1)
                        if aperture < aperturemax:
                            if aperture >= aperturemin:
                                # extract aperture taper value
                                aptap = (
                                    aptap
                                    * aperturetap[
                                        int(
                                            20 * ((aperture - aperturemin) // daperture)
                                        )
                                    ]
                                )
                            # time limit check
                            if 0 <= itravii < nt - 1:
                                # assign values
                                y[ii] += (
                                    (
                                        x[isrc * nr + irec, itravii] * (1 - travdii)
                                        + x[isrc * nr + irec, itravii + 1] * travdii
                                    )
                                    * damp
                                    * aptap
                                )
        return y

    def _register_multiplications(self, engine: str) -> None:
        if engine not in ["numpy", "numba", "cuda"]:
            raise KeyError("engine must be numpy or numba or cuda")
        if engine == "numba" and jit_message is None:
            # numba
            numba_opts = dict(
                nopython=True, nogil=True, parallel=parallel
            )  # fastmath=True,
            if self.dynamic and self.travsrcrec:
                self._kirch_matvec = jit(**numba_opts)(self._ampsrcrec_kirch_matvec)
                self._kirch_rmatvec = jit(**numba_opts)(self._ampsrcrec_kirch_rmatvec)
            elif self.travsrcrec:
                self._kirch_matvec = jit(**numba_opts)(self._travsrcrec_kirch_matvec)
                self._kirch_rmatvec = jit(**numba_opts)(self._travsrcrec_kirch_rmatvec)
            elif not self.travsrcrec:
                self._kirch_matvec = jit(**numba_opts)(self._trav_kirch_matvec)
                self._kirch_rmatvec = jit(**numba_opts)(self._trav_kirch_rmatvec)
        elif engine == "cuda":
            if self.dynamic and self.travsrcrec:
                self.cuda_helper = _kirchhoffCudaHelper(self.ns, self.nr, self.nt, self.ni, 1, 1)
                self.cuda_helper._data_prep_dynamic(self.ns, self.nr, self.nt, self.ni, self.nz, self.dt,
                                                    self.aperture, self.angleaperture,
                                                    self.aperturetap, self.vel, self.six, self.rix, self.trav_recs,
                                                    self.angle_recs, self.trav_srcs, self.angle_srcs,self.amp_srcs,
                                                    self.amp_recs)
            elif self.travsrcrec:
                self.cuda_helper = _kirchhoffCudaHelper(self.ns, self.nr, self.nt, self.ni, 0, 1)
            elif not self.travsrcrec:
                self.cuda_helper = _kirchhoffCudaHelper(self.ns, self.nr, self.nt, self.ni, 0, 0)
            self._kirch_matvec = self.cuda_helper._matvec_call
            self._kirch_rmatvec = self.cuda_helper._rmatvec_call
        else:
            if engine == "numba" and jit_message is not None:
                logging.warning(jit_message)
            if self.dynamic and self.travsrcrec:
                self._kirch_matvec = self._ampsrcrec_kirch_matvec
                self._kirch_rmatvec = self._ampsrcrec_kirch_rmatvec
            elif self.travsrcrec:
                self._kirch_matvec = self._travsrcrec_kirch_matvec
                self._kirch_rmatvec = self._travsrcrec_kirch_rmatvec
            elif not self.travsrcrec:
                self._kirch_matvec = self._trav_kirch_matvec
                self._kirch_rmatvec = self._trav_kirch_rmatvec

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = np.zeros((self.nsnr, self.nt), dtype=self.dtype)
        if self.dynamic and self.travsrcrec:
            inputs = (
                x.ravel(),
                y,
                self.ns,
                self.nr,
                self.nt,
                self.ni,
                self.dt,
                self.vel,
                self.trav_srcs,
                self.trav_recs,
                self.amp_srcs,
                self.amp_recs,
                self.aperture[0],
                self.aperture[1],
                self.aperturetap,
                self.nz,
                self.six,
                self.rix,
                self.angleaperture[0],
                self.angleaperture[1],
                self.angle_srcs,
                self.angle_recs,
            )
        elif self.travsrcrec:
            inputs = (
                x.ravel(),
                y,
                self.ns,
                self.nr,
                self.nt,
                self.ni,
                self.dt,
                self.trav_srcs,
                self.trav_recs,
            )
        elif not self.travsrcrec:
            inputs = (x.ravel(), y, self.nsnr, self.nt, self.ni, self.itrav, self.travd)

        y = self._kirch_matvec(*inputs)
        y = self.cop._matvec(y.ravel())
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        x = self.cop._rmatvec(x.ravel())
        x = x.reshape(self.nsnr, self.nt)
        y = np.zeros(self.ni, dtype=self.dtype)
        if self.dynamic and self.travsrcrec:
            inputs = (
                x,
                y,
                self.ns,
                self.nr,
                self.nt,
                self.ni,
                self.dt,
                self.vel,
                self.trav_srcs,
                self.trav_recs,
                self.amp_srcs,
                self.amp_recs,
                self.aperture[0],
                self.aperture[1],
                self.aperturetap,
                self.nz,
                self.six,
                self.rix,
                self.angleaperture[0],
                self.angleaperture[1],
                self.angle_srcs,
                self.angle_recs,
            )
        elif self.travsrcrec:
            inputs = (
                x,
                y,
                self.ns,
                self.nr,
                self.nt,
                self.ni,
                self.dt,
                self.trav_srcs,
                self.trav_recs,
            )
        elif not self.travsrcrec:
            inputs = (x, y, self.nsnr, self.nt, self.ni, self.itrav, self.travd)

        y = self._kirch_rmatvec(*inputs)
        return y
