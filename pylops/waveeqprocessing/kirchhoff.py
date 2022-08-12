import logging

import numpy as np

from pylops import LinearOperator
from pylops.signalprocessing import Convolve1D
from pylops.utils.decorators import reshaped

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
        Wavelet
    wavcenter : :obj:`int`
        Index of wavelet center
    y : :obj:`numpy.ndarray`
        Additional spatial axis (for 3-dimensional problems)
    trav : :obj:`numpy.ndarray`, optional
        Traveltime table of size
        :math:`\lbrack (n_y) n_x n_z \times n_s n_r \rbrack` (to be provided if
        ``mode='byot'``)
    dist : :obj:`numpy.ndarray`, optional
        Distance table of size
        :math:`\lbrack (n_y) n_x n_z \times n_s n_r \rbrack` (to be provided if
        ``mode='byot'``)
    mode : :obj:`str`, optional
        Computation mode (``analytic``, ``eikonal`` or ``byot``, see Notes for
        more details)
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
    The Kirchhoff demigration operator synthetises seismic data given a
    propagation velocity model :math:`v` and a reflectivity model :math:`m`.
    In forward mode:

    .. math::
        d(\mathbf{x_r}, \mathbf{x_s}, t) =
        w(t) * \int_V G(\mathbf{x_r}, \mathbf{x}, t)
        m(\mathbf{x}) G(\mathbf{x}, \mathbf{x_s}, t)\,\mathrm{d}\mathbf{x}

    where :math:`m(\mathbf{x})` is the model and it represents the reflectivity
    at every location in the subsurface, :math:`G(\mathbf{x}, \mathbf{x_s}, t)`
    and :math:`G(\mathbf{x_r}, \mathbf{x}, t)` are the Green's functions
    from source-to-subsurface-to-receiver and finally  :math:`w(t)` is the
    wavelet. In our current implementation, the following high-frequency
    approximation of the Green's functions is adopted:

    .. math::
        G(\mathbf{x_r}, \mathbf{x}, \omega) = \frac{1}{d(\mathbf{x_r}, \mathbf{x})}
            e^{j \omega t(\mathbf{x_r}, \mathbf{x})}

    where $d(\mathbf{x_r}, \mathbf{x})$ is the distance and
    $t(\mathbf{x_r}, \mathbf{x})$ is the traveltime.

    Depending on the choice of ``mode`` the Green's function will be
    computed and applied differently:

    * ``mode=analytic`` or ``mode=eikonal``: traveltime curves between
      source to receiver pairs are computed for every subsurface point and
      Green's functions are implemented from traveltime look-up tables, placing
      the reflectivity values at corresponding source-to-receiver time in the
      data.
    * ``byot``: bring your own table. Traveltime table provided
      directly by user using ``trav`` input parameter. Green's functions are
      then implemented in the same way as previous options.

    The adjoint of the demigration operator is a *migration* operator which
    projects data in the model domain creating an image of the subsurface
    reflectivity.

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
        trav=None,
        dist=None,
        mode="eikonal",
        engine="numpy",
        dtype="float64",
        name="D",
    ):
        ndim, _, dims, ny, nx, nz, ns, nr, _, _, _, _, _ = Kirchhoff._identify_geometry(
            z, x, srcs, recs, y=y
        )
        dt = t[1] - t[0]
        self.nt = len(t)

        if mode in ["analytic", "eikonal", "byot"]:
            if mode in ["analytic", "eikonal"]:
                # compute traveltime table
                self.trav, _, _, self.dist = Kirchhoff._traveltime_table(
                    z, x, srcs, recs, vel, y=y, mode=mode
                )
            else:
                self.trav = trav
                self.dist = dist
        else:
            raise NotImplementedError("method must be analytic or eikonal")

        self.dist += 1e-10 * np.max(self.dist)  # need to add to avoid division by 0
        self.itrav = (self.trav / dt).astype("int32")
        self.travd = self.trav / dt - self.itrav
        self.cop = Convolve1D(
            (ns * nr, self.nt), h=wav, offset=wavcenter, axis=1, dtype=dtype
        )
        self.nsnr = ns * nr
        self.ni = np.prod(dims)

        dims = tuple(dims) if ndim == 2 else (dims[0] * dims[1], dims[2])
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

        """
        (
            ndims,
            shiftdim,
            _,
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

        return trav, trav_srcs, trav_recs, dist

    @staticmethod
    def _kirch_matvec(x, y, nsnr, nt, ni, itrav, travd, dist):
        for isrcrec in prange(nsnr):
            itravisrcrec = itrav[:, isrcrec]
            travdisrcrec = travd[:, isrcrec]
            distisrcrec = dist[:, isrcrec]
            for ii in range(ni):
                index = itravisrcrec[ii]
                dindex = travdisrcrec[ii]
                ddist = distisrcrec[ii]
                if 0 <= index < nt - 1:
                    y[isrcrec, index] += x[ii] * (1 - dindex) / ddist
                    y[isrcrec, index + 1] += x[ii] * dindex / ddist
        return y

    @staticmethod
    def _kirch_rmatvec(x, y, nsnr, nt, ni, itrav, travd, dist):
        for ii in prange(ni):
            itravii = itrav[ii]
            travdii = travd[ii]
            distii = dist[ii]
            for isrcrec in range(nsnr):
                if 0 <= itravii[isrcrec] < nt - 1:
                    y[ii] += (
                        x[isrcrec, itravii[isrcrec]] * (1 - travdii[isrcrec])
                        + x[isrcrec, itravii[isrcrec] + 1] * travdii[isrcrec]
                    ) / distii[isrcrec]
        return y

    def _register_multiplications(self, engine):
        if engine not in ["numpy", "numba"]:
            raise KeyError("engine must be numpy or numba")
        if engine == "numba" and jit is not None:
            # numba
            numba_opts = dict(
                nopython=True, nogil=True, parallel=True
            )  # fastmath=True,
            self._kirch_matvec = jit(**numba_opts)(self._kirch_matvec)
            self._kirch_rmatvec = jit(**numba_opts)(self._kirch_rmatvec)
        else:
            if engine == "numba" and jit is None:
                logging.warning(jit_message)

    @reshaped
    def _matvec(self, x):
        y = np.zeros((self.nsnr, self.nt), dtype=self.dtype)
        y = self._kirch_matvec(
            x.ravel(), y, self.nsnr, self.nt, self.ni, self.itrav, self.travd, self.dist
        )
        y = self.cop._matvec(y.ravel())
        return y

    @reshaped
    def _rmatvec(self, x):
        x = self.cop._rmatvec(x.ravel())
        x = x.reshape(self.nsnr, self.nt)
        y = np.zeros(self.ni, dtype=self.dtype)
        y = self._kirch_rmatvec(
            x, y, self.nsnr, self.nt, self.ni, self.itrav, self.travd, self.dist
        )
        return y
