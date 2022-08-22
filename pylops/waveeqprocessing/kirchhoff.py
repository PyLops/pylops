import logging
import os

import numpy as np

from pylops import LinearOperator
from pylops.signalprocessing import Convolve1D
from pylops.utils.decorators import reshaped
from pylops.utils.tapers import taper

try:
    import skfmm
except ModuleNotFoundError:
    skfmm = None
    skfmm_message = (
        "Skfmm package not installed. Choose method=analytical "
        "if using constant velocity or run "
        '"pip install scikit-fmm" or '
        '"conda install -c conda-forge scikit-fmm".'
    )
except Exception as e:
    skfmm = None
    skfmm_message = f"Failed to import skfmm (error:{e})."

try:
    from numba import jit, prange

    # detect whether to use parallel or not
    numba_threads = int(os.getenv("NUMBA_NUM_THREADS", "1"))
    parallel = True if numba_threads != 1 else False
except ModuleNotFoundError:
    jit = None
    prange = range
    jit_message = "Numba not available, reverting to numpy."
except Exception as e:
    jit = None
    jit_message = "Failed to import numba (error:%s), use numpy." % e


logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class Kirchhoff(LinearOperator):
    r"""Kirchhoff Demigration operator.

    Kirchhoff-based demigration/migration operator. Uses a high-frequency
    approximation of  Green's function propagators based on ``trav``.

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
    dynamic : :obj:`bool`, optional
        .. versionadded:: 2.0.0

        Include dynamic weights in computations (``True``) or not (``False``).
        Note that when ``mode=byot``, the user is required to provide such weights
        in ``amp``.
    trav : :obj:`numpy.ndarray`, optional
        Traveltime table of size
        :math:`\lbrack (n_y) n_x n_z \times n_s n_r \rbrack` (to be provided if
        ``mode='byot'``)
    amp : :obj:`numpy.ndarray`, optional
        .. versionadded:: 2.0.0

        Amplitude table of size
        :math:`\lbrack (n_y) n_x n_z \times n_s n_r \rbrack` (to be provided if
        ``mode='byot'``)
    aperture : :obj:`float`, optional
        .. versionadded:: 2.0.0

        Maximum allowed aperture expressed as the ratio of offset over depth
    aperturetap : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Size of hanning taper to apply at the edges of the aperture
    angleaperture : :obj:`float`, optional
        .. versionadded:: 2.0.0

        Maximum allowed angle (either source or receiver side) in degrees
    snell : :obj:`float`, optional
        .. versionadded:: 2.0.0

        Threshold of Snell's law evaluation. If larger, the source-receiver-image
        point is discarded. If ``None``, no check on the validiyy of the Snell's
        law is performed
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
    propagation velocity model :math:`v` and a reflectivity model :math:`m`.
    In forward mode [1]_, [2]_:

    .. math::
        d(\mathbf{x_r}, \mathbf{x_s}, t) =
        \tilde{w}(t) * \int_V \frac{ \partial ( G(\mathbf{x_r}, \mathbf{x}, t)
         G(\mathbf{x}, \mathbf{x_s}, t))}{\partial n}  m(\mathbf{x}) \,\mathrm{d}\mathbf{x}

    where :math:`m(\mathbf{x})` represents the reflectivity
    at every location in the subsurface, :math:`G(\mathbf{x}, \mathbf{x_s}, t)`
    and :math:`G(\mathbf{x_r}, \mathbf{x}, t)` are the Green's functions
    from source-to-subsurface-to-receiver and finally :math:`\tilde{w}(t)` is
    a filtered version of the wavelet :math:`w(t)`.  In our implementation,
    the following high-frequency approximation of the Green's functions is adopted:

    .. math::
        G(\mathbf{x_r}, \mathbf{x}, \omega) = a(\mathbf{x_r}, \mathbf{x})
            e^{j \omega t(\mathbf{x_r}, \mathbf{x})}

    where :math:`a(\mathbf{x_r}, \mathbf{x})` is the amplitude and
    :math:`t(\mathbf{x_r}, \mathbf{x})` is the traveltime. When ``dynamic=False`` the
    amplitude is disregarded leading to a kinematic-only Kirchhoff operator.

    .. math::
        d(\mathbf{x_r}, \mathbf{x_s}, t) =
        \tilde{w}(t) * \int_V e^{j \omega (t(\mathbf{x_r}, \mathbf{x}) +
        t(\mathbf{x}, \mathbf{x_s}))} m(\mathbf{x}) \,\mathrm{d}\mathbf{x}

    On the  other hand, when ``dynamic=True``, the amplitude scaling is defined as
    :math:`a(\mathbf{x_r}, \mathbf{x})=1/d(\mathbf{x_r}, \mathbf{x})`, where ``d`` is
    the distance between the two points and represents the geometrical spreading
    of the wavefront. Moreover an angle scaling is included in the modelling operator
    added as follows:

    .. math::
        d(\mathbf{x_r}, \mathbf{x_s}, t) =
        \tilde{w}(t) * \int_V a(\mathbf{x}, \mathbf{x_r}, \mathbf{x_s})
        \frac{(cos \theta_s + cos \theta_r)} {v(\mathbf{x})} e^{j \omega (t(\mathbf{x_r}, \mathbf{x}) +
         t(\mathbf{x}, \mathbf{x_s}))} m(\mathbf{x}) \,\mathrm{d}\mathbf{x}

    where :math:`\theta_s` and :math:`\theta_r` are the angles between the source-side
    and receiver-side rays and the normal to the reflector  at the image point (or
    the vertical axis at the image point when ``reflslope=None``), respectively.

    Depending on the choice of ``mode`` the traveltime and amplitude of the Green's
    function will be also computed differently:

    * ``mode=analytic`` or ``mode=eikonal``: traveltimes, geometrical spreading, and angles
      are computed for every source-image point-receiver triplets and the
      Green's functions are implemented from traveltime look-up tables, placing
      scaled reflectivity values at corresponding source-to-receiver time in the
      data.
    * ``byot``: bring your own tables. Traveltime table are provided
      directly by user using ``trav`` input parameter. Similarly, in this case one
      can provide their own amplitude scaling ``amp`` (which should include the angle
      scaling too).

    Finally, an aperture limitation is also implemented as defined by ``aperture``
    and a taper is added at the edges of the aperture.

    The adjoint of the demigration operator is a *migration* operator which
    projects data in the model domain creating an image of the subsurface
    reflectivity.

    .. [1] Bleistein, N., Cohen, J.K., and Stockwell, J.W..
       "Mathematics of Multidimensional Seismic Imaging, Migration and
       Inversion", 2000.

    .. [2] Santos, L.T., Schleicher, J., Tygel, M., and Hubral, P.
       "Seismic modeling by demigration", Geophysics, 65(4), pp. 1281-1289, 2000.

    """

    def __init__(
        self,
        z,
        x,
        t,
        srcs,
        recs,
        vel,
        wav,
        wavcenter,
        y=None,
        mode="eikonal",
        dynamic=False,
        trav=None,
        amp=None,
        aperture=None,
        aperturetap=20,
        angleaperture=90,
        sloperefl=None,
        snell=None,
        engine="numpy",
        dtype="float64",
        name="D",
    ):
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
        dt = t[1] - t[0]
        self.nt = len(t)

        # store ix-iy locations of sources and receivers
        dx = x[1] - x[0]
        if self.ndims == 2:
            self.six = np.tile((srcs[0] - x[0]) // dx, (nr, 1)).T.astype(int).ravel()
            self.rix = np.tile((recs[0] - x[0]) // dx, (ns, 1)).astype(int).ravel()

        # compute traveltime
        self.dynamic = dynamic
        if mode in ["analytic", "eikonal", "byot"]:
            if mode in ["analytic", "eikonal"]:
                # compute traveltime table
                (
                    self.trav,
                    trav_srcs,
                    trav_recs,
                    dist,
                    trav_srcs_grad,
                    trav_recs_grad,
                ) = Kirchhoff._traveltime_table(z, x, srcs, recs, vel, y=y, mode=mode)
                if self.dynamic:
                    # need to add a scalar in the denominator to avoid division by 0
                    # currently set to 1/100 of max distance to avoid having to large
                    # scaling around the source. This number may change in future or
                    # left to the user to define
                    epsdist = 1e-2
                    self.amp = 1 / (dist + epsdist * np.max(dist))

                    # compute angles
                    if self.ndims == 2:
                        # 2d with vertical
                        self.angle_srcs = np.arctan2(
                            trav_srcs_grad[0], trav_srcs_grad[1]
                        ).reshape(np.prod(dims), ns)
                        self.angle_recs = np.arctan2(
                            trav_recs_grad[0], trav_recs_grad[1]
                        ).reshape(np.prod(dims), nr)
                        self.cosangle = np.cos(self.angle_srcs).reshape(
                            np.prod(dims), ns, 1
                        ) + np.cos(self.angle_recs).reshape(np.prod(dims), 1, nr)
                        self.cosangle = self.cosangle.reshape(np.prod(dims), ns * nr)
                        self.amp *= self.cosangle
                        if mode == "analytic":
                            self.amp /= vel
                        else:
                            self.amp /= vel.reshape(np.prod(dims), 1)
                        # TODO: 2d with normal
                    else:
                        # TODO: 3D
                        raise NotImplementedError(
                            "dynamic=True currently not available in 3D"
                        )
            else:
                self.trav = trav
                if self.dynamic:
                    self.amp = amp

        else:
            raise NotImplementedError("method must be analytic or eikonal")

        self.itrav = (self.trav / dt).astype("int32")
        self.travd = self.trav / dt - self.itrav

        # create wavelet operator
        self.wav = self._wavelet_reshaping(wav, dt)
        self.cop = Convolve1D(
            (ns * nr, self.nt), h=wav, offset=wavcenter, axis=1, dtype=dtype
        )

        # define aperture and create aperture taper (of size 100)
        self.aperture = self.nx / self.nz if aperture is None else aperture
        self.aperturetap = taper(201, aperturetap, "hanning")[101:]

        # define angle aperture and snell law
        self.angleaperture = np.deg2rad(angleaperture)
        self.snell = np.pi if snell is None else np.deg2rad(snell)

        # dimensions
        self.nsnr = ns * nr
        self.ni = np.prod(dims)
        dims = tuple(dims) if self.ndims == 2 else (dims[0] * dims[1], dims[2])
        dimsd = (ns, nr, self.nt)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)
        self._register_multiplications(engine)

    @staticmethod
    def _identify_geometry(z, x, srcs, recs, y=None):
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
    def _traveltime_table(z, x, srcs, recs, vel, y=None, mode="eikonal"):
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
        trav : :obj:`numpy.ndarray`
            Total traveltime table of size :math:`\lbrack (n_y) n_x n_z
            \times n_s n_r \rbrack`
        trav_srcs : :obj:`numpy.ndarray`
            Source-to-subsurface traveltime table of size
            :math:`\lbrack (n_y*) n_x n_z \times n_s \rbrack` (or constant)
        trav_recs : :obj:`numpy.ndarray`
            Receiver-to-subsurface traveltime table of size
            :math:`\lbrack (n_y) n_x n_z \times n_r \rbrack`
        dist : :obj:`numpy.ndarray`
            Total distance table of size
            :math:`\lbrack (n_y*) n_x n_z \times n_s \rbrack` (or constant)
        trav_srcs_gradient : :obj:`numpy.ndarray`
            Source-to-subsurface traveltime gradient table of size
            :math:`\lbrack (n_y*) n_x n_z \times n_s \rbrack` (or constant)
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

            trav = trav_srcs.reshape(ny * nx * nz, ns, 1) + trav_recs.reshape(
                ny * nx * nz, 1, nr
            )
            trav = trav.reshape(ny * nx * nz, ns * nr)
            dist = trav * vel

        elif mode == "eikonal":
            if skfmm is not None:
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
                dist = dist_srcs.reshape(ny * nx * nz, ns, 1) + dist_recs.reshape(
                    ny * nx * nz, 1, nr
                )
                dist = dist.reshape(ny * nx * nz, ns * nr)
                trav = trav_srcs.reshape(ny * nx * nz, ns, 1) + trav_recs.reshape(
                    ny * nx * nz, 1, nr
                )
                trav = trav.reshape(ny * nx * nz, ns * nr)
            else:
                raise NotImplementedError(skfmm_message)
        else:
            raise NotImplementedError("method must be analytic or eikonal")

        # compute traveltime gradients at image points
        trav_srcs_grad = np.gradient(
            trav_srcs.reshape(*dims, ns), axis=np.arange(ndims)
        )
        trav_recs_grad = np.gradient(
            trav_recs.reshape(*dims, nr), axis=np.arange(ndims)
        )

        return trav, trav_srcs, trav_recs, dist, trav_srcs_grad, trav_recs_grad

    def _wavelet_reshaping(self, wav, dt):
        """Apply wavelet reshaping as from theory in [1]_"""
        f = np.fft.rfftfreq(len(wav), dt)
        W = np.fft.rfft(wav, n=len(wav))
        if self.ndims == 2:
            Wfilt = W * (2 * np.pi * f)
        else:
            Wfilt = W * (-1j * 2 * np.pi * f)
        wavfilt = np.fft.irfft(Wfilt, n=len(wav))
        return wavfilt

    @staticmethod
    def _trav_kirch_matvec(x, y, nsnr, nt, ni, itrav, travd):
        for isrcrec in prange(nsnr):
            itravisrcrec = itrav[:, isrcrec]
            travdisrcrec = travd[:, isrcrec]
            for ii in range(ni):
                index = itravisrcrec[ii]
                dindex = travdisrcrec[ii]
                if 0 <= index < nt - 1:
                    y[isrcrec, index] += x[ii] * (1 - dindex)
                    y[isrcrec, index + 1] += x[ii] * dindex
        return y

    @staticmethod
    def _trav_kirch_rmatvec(x, y, nsnr, nt, ni, itrav, travd):
        for ii in prange(ni):
            itravii = itrav[ii]
            travdii = travd[ii]
            for isrcrec in range(nsnr):
                if 0 <= itravii[isrcrec] < nt - 1:
                    y[ii] += (
                        x[isrcrec, itravii[isrcrec]] * (1 - travdii[isrcrec])
                        + x[isrcrec, itravii[isrcrec] + 1] * travdii[isrcrec]
                    )
        return y

    @staticmethod
    def _amp_kirch_matvec(
        x,
        y,
        nsnr,
        nt,
        ni,
        itrav,
        travd,
        amp,
        aperturemax,
        aperturetap,
        nz,
        six,
        rix,
        angleaperture,
        angles_srcs,
        angles_recs,
        snell,
    ):
        # TODO: NEED TO ADD Y FOR 3D CASE
        ns, nr = angles_srcs.shape[-1], angles_recs.shape[-1]
        for isrcrec in prange(nsnr):
            # extract traveltime, amplitude, src/rec coordinates at given src/pair
            itravisrcrec = itrav[:, isrcrec]
            travdisrcrec = travd[:, isrcrec]
            ampisrcrec = amp[:, isrcrec]
            sixisrcrec = six[isrcrec]
            rixisrcrec = rix[isrcrec]
            # extract source and receiver angles
            angles_src = angles_srcs[:, isrcrec // nr]
            angles_rec = angles_recs[:, isrcrec % nr]
            for ii in range(ni):
                # extract traveltime, amplitude at given image point
                index = itravisrcrec[ii]
                dindex = travdisrcrec[ii]
                damp = ampisrcrec[ii]
                # extract source and receiver angle
                angle_src = angles_src[ii]
                angle_rec = angles_rec[ii]
                if (
                    abs(angle_src) < angleaperture
                    and abs(angle_rec) < angleaperture
                    and abs(angle_src + angle_rec) < snell
                ):
                    # identify x-index of image point
                    iz = ii % nz
                    # aperture check
                    aperture = abs(sixisrcrec - rixisrcrec) / iz
                    if aperture < aperturemax:
                        # extract aperture taper value
                        aptap = aperturetap[int(100 * aperture // aperturemax)]
                        # time limit check
                        if 0 <= index < nt - 1:
                            # assign values
                            y[isrcrec, index] += x[ii] * (1 - dindex) * damp * aptap
                            y[isrcrec, index + 1] += x[ii] * dindex * damp * aptap
        return y

    @staticmethod
    def _amp_kirch_rmatvec(
        x,
        y,
        nsnr,
        nt,
        ni,
        itrav,
        travd,
        amp,
        aperturemax,
        aperturetap,
        nz,
        six,
        rix,
        angleaperture,
        angles_srcs,
        angles_recs,
        snell,
    ):
        # TODO: NEED TO ADD Y FOR 3D CASE
        ns, nr = angles_srcs.shape[-1], angles_recs.shape[-1]
        for ii in prange(ni):
            itravii = itrav[ii]
            travdii = travd[ii]
            ampii = amp[ii]
            # extract source and receiver angles
            angle_srcs = angles_srcs[ii]
            angle_recs = angles_recs[ii]
            # identify x-index of image point
            iz = ii % nz
            for isrcrec in range(nsnr):
                index = itravii[isrcrec]
                dindex = travdii[isrcrec]
                sixisrcrec = six[isrcrec]
                rixisrcrec = rix[isrcrec]
                # extract source and receiver angle
                angle_src = angle_srcs[isrcrec // nr]
                angle_rec = angle_recs[isrcrec % nr]
                if (
                    abs(angle_src) < angleaperture
                    and abs(angle_rec) < angleaperture
                    and abs(angle_src + angle_rec) < snell
                ):
                    # aperture check
                    aperture = abs(sixisrcrec - rixisrcrec) / iz
                    if aperture < aperturemax:
                        # extract aperture taper value
                        aptap = aperturetap[int(100 * aperture // aperturemax)]
                        # time limit check
                        if 0 <= index < nt - 1:
                            # assign values
                            y[ii] += (
                                (
                                    x[isrcrec, index] * (1 - dindex)
                                    + x[isrcrec, index + 1] * dindex
                                )
                                * ampii[isrcrec]
                                * aptap
                            )
        return y

    def _register_multiplications(self, engine):
        if engine not in ["numpy", "numba"]:
            raise KeyError("engine must be numpy or numba")
        if engine == "numba" and jit is not None:
            # numba
            numba_opts = dict(
                nopython=True, nogil=True, parallel=parallel
            )  # fastmath=True,
            if self.dynamic:
                self._kirch_matvec = jit(**numba_opts)(self._amp_kirch_matvec)
                self._kirch_rmatvec = jit(**numba_opts)(self._amp_kirch_rmatvec)
            else:
                self._kirch_matvec = jit(**numba_opts)(self._trav_kirch_matvec)
                self._kirch_rmatvec = jit(**numba_opts)(self._trav_kirch_rmatvec)
        else:
            if engine == "numba" and jit is None:
                logging.warning(jit_message)
            if self.dynamic:
                self._kirch_matvec = self._amp_kirch_matvec
                self._kirch_rmatvec = self._amp_kirch_rmatvec
            else:
                self._kirch_matvec = self._trav_kirch_matvec
                self._kirch_rmatvec = self._trav_kirch_rmatvec

    @reshaped
    def _matvec(self, x):
        y = np.zeros((self.nsnr, self.nt), dtype=self.dtype)
        if self.dynamic:
            inputs = (
                x.ravel(),
                y,
                self.nsnr,
                self.nt,
                self.ni,
                self.itrav,
                self.travd,
                self.amp,
                self.aperture,
                self.aperturetap,
                self.nz,
                self.six,
                self.rix,
                self.angleaperture,
                self.angle_srcs,
                self.angle_recs,
                self.snell,
            )
        else:
            inputs = (x.ravel(), y, self.nsnr, self.nt, self.ni, self.itrav, self.travd)
        y = self._kirch_matvec(*inputs)
        y = self.cop._matvec(y.ravel())
        return y

    @reshaped
    def _rmatvec(self, x):
        x = self.cop._rmatvec(x.ravel())
        x = x.reshape(self.nsnr, self.nt)
        y = np.zeros(self.ni, dtype=self.dtype)
        if self.dynamic:
            inputs = (
                x,
                y,
                self.nsnr,
                self.nt,
                self.ni,
                self.itrav,
                self.travd,
                self.amp,
                self.aperture,
                self.aperturetap,
                self.nz,
                self.six,
                self.rix,
                self.angleaperture,
                self.angle_srcs,
                self.angle_recs,
                self.snell,
            )
        else:
            inputs = (x, y, self.nsnr, self.nt, self.ni, self.itrav, self.travd)
        y = self._kirch_rmatvec(*inputs)
        return y
