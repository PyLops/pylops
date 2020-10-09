import logging
import numpy as np

from scipy.sparse.linalg import lsqr
from pylops import Spread
from pylops.signalprocessing import Convolve1D
from pylops.utils import dottest as Dottest

try:
    import skfmm
except ModuleNotFoundError:
    skfmm = None
    skfmm_message = 'Skfmm package not installed. Choose method=analytical ' \
                    'if using constant velocity or run ' \
                    '"pip install scikit-fmm" or ' \
                    '"conda install -c conda-forge scikit-fmm".'
except Exception as e:
    skfmm = None
    skfmm_message = 'Failed to import skfmm (error:%s).' % e

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


def _identify_geometry(z, x, srcs, recs, y=None):
    """Identify geometry and acquisition size and sampling
    """
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
    return ndims, shiftdim, dims, ny, nx, nz, ns, nr, dy, dx, dz, \
           dsamp, origin


def _traveltime_table(z, x, srcs, recs, vel, y=None, mode='eikonal'):
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
        Sources in array of size :math:`\lbrack 2/3 \times n_s \rbrack`
    recs : :obj:`numpy.ndarray`
        Receivers in array of size :math:`\lbrack 2/3 \times n_r \rbrack`
    vel : :obj:`numpy.ndarray` or :obj:`float`
        Velocity model of size :math:`\lbrack (n_y \times) n_x
        \times n_z \rbrack` (or constant)
    y : :obj:`numpy.ndarray`
        Additional spatial axis (for 3-dimensional problems)
    mode : :obj:`numpy.ndarray`, optional
        Computation mode (``eikonal``, ``analytic`` - only for constant velocity)

    Returns
    -------
    trav : :obj:`numpy.ndarray`
        Total traveltime table of size :math:`\lbrack (n_y*) n_x*n_z
        \times n_s*n_r \rbrack`
    trav_srcs : :obj:`numpy.ndarray`
        Source-to-subsurface traveltime table of size
        :math:`\lbrack (n_y*) n_x*n_z \times n_s \rbrack` (or constant)
    trav_recs : :obj:`numpy.ndarray`
        Receiver-to-subsurface traveltime table of size
        :math:`\lbrack (n_y*) n_x*n_z \times n_r \rbrack`

    """
    ndims, shiftdim, _, ny, nx, nz, ns, nr, _, _, _, dsamp, origin = \
        _identify_geometry(z, x, srcs, recs, y=y)
    if mode == 'analytic':
        if not isinstance(vel, (float, int)):
            raise ValueError('vel must be scalar for mode=analytical')

        # compute grid
        if ndims == 2:
            X, Z = np.meshgrid(x, z, indexing='ij')
            X, Z = X.flatten(), Z.flatten()
        else:
            Y, X, Z = np.meshgrid(y, x, z, indexing='ij')
            Y, X, Z = Y.flatten(), X.flatten(), Z.flatten()

        dist_srcs2 = np.zeros((ny * nx * nz, ns))
        dist_recs2 = np.zeros((ny * nx * nz, nr))
        for isrc, src in enumerate(srcs.T):
            dist_srcs2[:, isrc] = (X - src[0+ shiftdim]) ** 2 + \
                                  (Z - src[1+ shiftdim]) ** 2
            if ndims == 3:
                dist_srcs2[:, isrc] += (Y - src[0]) ** 2
        for irec, rec in enumerate(recs.T):
            dist_recs2[:, irec] = (X - rec[0 + shiftdim]) ** 2 + \
                                  (Z - rec[1 + shiftdim]) ** 2
            if ndims == 3:
                dist_recs2[:, irec] += (Y - rec[0]) ** 2
        trav_srcs = np.sqrt(dist_srcs2) / vel
        trav_recs = np.sqrt(dist_recs2) / vel

        trav = trav_srcs.reshape(ny * nx * nz, ns, 1) + \
               trav_recs.reshape(ny * nx * nz, 1, nr)
        trav = trav.reshape(ny * nx * nz, ns * nr)

    elif mode == 'eikonal':
        if skfmm is not None:
            trav_srcs = np.zeros((ny * nx * nz, ns))
            trav_recs = np.zeros((ny * nx * nz, nr))
            for isrc, src in enumerate(srcs.T):
                src = np.round((src-origin)/dsamp).astype(np.int32)
                phi = np.ones_like(vel)
                if ndims == 2:
                    phi[src[0], src[1]] = -1
                else:
                    phi[src[0], src[1], src[2]] = -1
                trav_srcs[:, isrc] = (skfmm.travel_time(phi=phi,
                                                        speed=vel,
                                                        dx=dsamp)).ravel()
            for irec, rec in enumerate(recs.T):
                rec = np.round((rec-origin)/dsamp).astype(np.int32)
                phi = np.ones_like(vel)
                if ndims == 2:
                    phi[rec[0], rec[1]] = -1
                else:
                    phi[rec[0], rec[1], rec[2]] = -1
                trav_recs[:, irec] = (skfmm.travel_time(phi=phi,
                                                        speed=vel,
                                                        dx=dsamp)).ravel()
            trav = trav_srcs.reshape(ny * nx * nz, ns, 1) + \
                   trav_recs.reshape(ny * nx * nz, 1, nr)
            trav = trav.reshape(ny * nx * nz, ns * nr)
        else:
            raise NotImplementedError(skfmm_message)
    else:
        raise NotImplementedError('method must be analytic or eikonal')

    return trav, trav_srcs, trav_recs


def Demigration(z, x, t, srcs, recs, vel, wav, wavcenter,
                y=None, trav=None, mode='eikonal'):
    r"""Kirchoff Demigration operator.

    Traveltime based seismic demigration/migration operator.

    Parameters
    ----------
    z : :obj:`numpy.ndarray`
        Depth axis
    x : :obj:`numpy.ndarray`
        Spatial axis
    t : :obj:`numpy.ndarray`
        Time axis for data
    srcs : :obj:`numpy.ndarray`
        Sources in array of size :math:`\lbrack 2/3 \times n_s \rbrack`
        The first axis should be ordered as (``y``,) ``x``, ``z``.
    recs : :obj:`numpy.ndarray`
        Receivers in array of size :math:`\lbrack 2/3 \times n_r \rbrack`
        The first axis should be ordered as (``y``,) ``x``, ``z``.
    vel : :obj:`numpy.ndarray` or :obj:`float`
        Velocity model of size :math:`\lbrack (n_y \times) n_x
        \times n_z \rbrack` (or constant)
    wav : :obj:`numpy.ndarray`
        Wavelet
    wavcenter : :obj:`int`
        Index of wavelet center
    y : :obj:`numpy.ndarray`
        Additional spatial axis (for 3-dimensional problems)
    mode : :obj:`str`, optional
        Computation mode (``analytic``, ``eikonal`` or ``byot``, see Notes for
        more details)
    trav : :obj:`numpy.ndarray`, optional
        Traveltime table of size
        :math:`\lbrack (n_y*) n_x*n_z \times n_r \rbrack` (to be provided if
        ``mode='byot'``)

    Returns
    -------
    demop : :obj:`pylops.LinearOperator`
        Demigration/Migration operator

    Raises
    ------
    NotImplementedError
        If ``mode`` is neither ``analytic``, ``eikonal``, or ``byot``

    Notes
    -----
    The demigration operator synthetizes seismic data given from a propagation
    velocity model :math:`v` and a reflectivity model :math:`m`. In forward
    mode:

    .. math::
        d(\mathbf{x_r}, \mathbf{x_s}, t) =
        w(t) * \int_V G(\mathbf{x}, \mathbf{x_s}, t)
        G(\mathbf{x_r}, \mathbf{x}, t) m(\mathbf{x}) d\mathbf{x}

    where :math:`m(\mathbf{x})` is the model and it represents the reflectivity
    at every location in the subsurface, :math:`G(\mathbf{x}, \mathbf{x_s}, t)`
    and :math:`G(\mathbf{x_r}, \mathbf{x}, t)` are the Green's functions
    from source-to-subsurface-to-receiver and finally  :math:`w(t)` is the
    wavelet. Depending on the choice of ``mode`` the Green's function will be
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
    ndim, _, dims, ny, nx, nz, ns, nr, _, _, _, _, _ = \
        _identify_geometry(z, x, srcs, recs, y=y)
    dt = t[1] - t[0]
    nt = len(t)
    if mode in ['analytic', 'eikonal', 'byot']:
        if mode in ['analytic', 'eikonal']:
            # compute traveltime table
            trav = _traveltime_table(z, x, srcs, recs, vel, y=y, mode=mode)[0]

        itrav = (trav / dt).astype('int32')
        travd = (trav / dt - itrav)
        if ndim == 2:
            itrav = itrav.reshape(nx, nz, ns * nr)
            travd = travd.reshape(nx, nz, ns * nr)
            dims = tuple(dims)
        else:
            itrav = itrav.reshape(ny*nx, nz, ns * nr)
            travd = travd.reshape(ny*nx, nz, ns * nr)
            dims = (dims[0]*dims[1], dims[2])

        # create operator
        sop = Spread(dims=dims, dimsd=(ns * nr, nt),
                     table=itrav, dtable=travd, engine='numba')

        cop = Convolve1D(ns * nr * nt, h=wav, offset=wavcenter,
                         dims=(ns * nr, nt),
                         dir=1)
        demop = cop * sop
    else:
        raise NotImplementedError('method must be analytic or eikonal')
    return demop


class LSM():
    r"""Least-squares Migration (LSM).

    Solve seismic migration as inverse problem given smooth velocity model
    ``vel`` and an acquisition setup identified by sources (``src``) and
    receivers (``recs``)

    Parameters
    ----------
    z : :obj:`numpy.ndarray`
        Depth axis
    x : :obj:`numpy.ndarray`
        Spatial axis
    t : :obj:`numpy.ndarray`
        Time axis for data
    srcs : :obj:`numpy.ndarray`
        Sources in array of size :math:`\lbrack 2/3 \times n_s \rbrack`
    recs : :obj:`numpy.ndarray`
        Receivers in array of size :math:`\lbrack 2/3 \times n_r \rbrack`
    vel : :obj:`numpy.ndarray` or :obj:`float`
        Velocity model of size :math:`\lbrack (n_y \times) n_x
        \times n_z \rbrack` (or constant)
    wav : :obj:`numpy.ndarray`
        Wavelet
    wavcenter : :obj:`int`
        Index of wavelet center
    y : :obj:`numpy.ndarray`
        Additional spatial axis (for 3-dimensional problems)
    mode : :obj:`numpy.ndarray`, optional
        Computation mode (``eikonal``, ``analytic`` - only for
        constant velocity)
    dottest : :obj:`bool`, optional
        Apply dot-test

    Attributes
    ----------
    Demop : :class:`pylops.LinearOperator`
        Demigration operator

    See Also
    --------
    pylops.waveeqprocessing.Demigration : Demigration operator

    Notes
    -----
    Inverting a demigration operator is generally referred in the literature
    as least-squares migration (LSM) as historically a least-squares cost
    function has been used for this purpose. In practice any other cost
    function could be used, for examples if
    ``solver='pylops.optimization.sparsity.FISTA'`` a sparse representation of
    reflectivity is produced as result of the inversion.

    Finally, it is worth noting that in the first iteration of an iterative
    scheme aimed at inverting the demigration operator, a projection of the
    recorded data in the model domain is performed and an approximate
    (band-limited)  image of the subsurface is created. This process is
    referred to in the literature as *migration*.

    """
    def __init__(self, z, x, t, srcs, recs, vel, wav, wavcenter, y=None,
                 mode='eikonal', dottest=False):
        self.y, self.x, self.z = y, x, z
        self.Demop = Demigration(z, x, t, srcs, recs, vel, wav, wavcenter,
                                 y=y, mode=mode)
        if dottest:
            Dottest(self.Demop, self.Demop.shape[0], self.Demop.shape[1],
                    raiseerror=True, verb=True)

    def solve(self, d, solver=lsqr, **kwargs_solver):
        r"""Solve least-squares migration equations with chosen ``solver``

        Parameters
        ----------
        d : :obj:`numpy.ndarray`
            Input data of size :math:`\lbrack n_s \times n_r
            \times n_t \rbrack`
        solver : :obj:`func`, optional
            Solver to be used for inversion
        **kwargs_solver
            Arbitrary keyword arguments for chosen ``solver``

        Returns
        -------
        minv : :obj:`np.ndarray`
            Inverted reflectivity model of size :math:`\lbrack (n_y \times)
            n_x \times n_z \rbrack`

        """
        minv = solver(self.Demop, d.ravel(), **kwargs_solver)[0]

        if self.y is None:
            minv = minv.reshape(len(self.x), len(self.z))
        else:
            minv = minv.reshape(len(self.y), len(self.x), len(self.z))

        return minv
