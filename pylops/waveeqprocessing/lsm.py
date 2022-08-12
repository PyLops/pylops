import logging

from scipy.sparse.linalg import lsqr

from pylops.utils import dottest as Dottest
from pylops.waveeqprocessing.kirchhoff import Kirchhoff
from pylops.waveeqprocessing.twoway import AcousticWave2D

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

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class LSM:
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
        Sources in array of size :math:`\lbrack 2(3) \times n_s \rbrack`
    recs : :obj:`numpy.ndarray`
        Receivers in array of size :math:`\lbrack 2(3) \times n_r \rbrack`
    vel : :obj:`numpy.ndarray` or :obj:`float`
        Velocity model of size :math:`\lbrack (n_y \times)\, n_x
        \times n_z \rbrack` (or constant)
    wav : :obj:`numpy.ndarray`
        Wavelet
    wavcenter : :obj:`int`
        Index of wavelet center
    y : :obj:`numpy.ndarray`
        Additional spatial axis (for 3-dimensional problems)
    kind : :str`, optional
        Kind of modelling operator (``kirchhoff``, ``twowayac``)
    mode : :obj:`str`, optional
        Computation mode (``eikonal``, ``analytic`` - only for
        constant velocity)
    engine : :obj:`str`, optional
        Engine used for computations (``numpy`` or ``numba``) when ``kind=kirchhoff``
        is used
    dottest : :obj:`bool`, optional
        Apply dot-test

    Attributes
    ----------
    Demop : :class:`pylops.LinearOperator`
        Demigration operator operator

    See Also
    --------
    pylops.waveeqprocessing.Kirchhoff : Kirchhoff operator
    pylops.waveeqprocessing.AcousticWave2D : AcousticWave2D operator

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
        kind="kirchhoff",
        mode="eikonal",
        engine="numba",
        dottest=False,
    ):
        self.y, self.x, self.z = y, x, z

        if kind == "kirchhoff":
            self.Demop = Kirchhoff(
                z,
                x,
                t,
                srcs,
                recs,
                vel,
                wav,
                wavcenter,
                y=y,
                mode=mode,
                engine=engine,
            )
        elif kind == "twowayac":
            shape = (len(x), len(z))
            origin = (x[0], z[0])
            spacing = (x[1] - x[0], z[1] - z[0])
            self.Demop = AcousticWave2D(
                shape,
                origin,
                spacing,
                vel,
                srcs[0],
                srcs[1],
                recs[0],
                recs[1],
                t[0],
                len(t),
            )

        else:
            raise NotImplementedError("kind must be kirchhoff or twowayac")

        if dottest:
            Dottest(
                self.Demop,
                self.Demop.shape[0],
                self.Demop.shape[1],
                raiseerror=True,
                verb=True,
            )

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
