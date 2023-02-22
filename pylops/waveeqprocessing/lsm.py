__all__ = ["LSM"]

import logging
from typing import Callable, Optional

from scipy.sparse.linalg import lsqr

from pylops.utils import dottest as Dottest
from pylops.utils.typing import NDArray
from pylops.waveeqprocessing.kirchhoff import Kirchhoff
from pylops.waveeqprocessing.twoway import AcousticWave2D

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class LSM:
    r"""Least-squares Migration (LSM).

    Solve seismic migration as inverse problem given smooth velocity model
    ``vel`` and an acquisition setup identified by sources (``src``) and
    receivers (``recs``).

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
        Kind of modelling operator (``kirchhoff``, ``twoway``)
    dottest : :obj:`bool`, optional
        Apply dot-test
    **kwargs_mod : :obj:`int`, optional
        Additional arguments to pass to modelling operators

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

    This routines provides users with a easy-to-use, out-of-the-box least-squares
    migration application that currently implements:

    - Kirchhoff LSM: this problem is parametrized in terms of reflectivity
      (i.e., vertical derivative of the acoustic impedance - or velocity in case of
      constant density). Currently, a ray-based modelling engine is used for this case
      (see :class:`pylops.waveeqprocessing.Kirchhoff`).

    - Born LSM: this problem is parametrized in terms of squared slowness perturbation
      (in the constant density case) and it is solved using an acoustic two-way eave equation
      modelling engine (see :class:`pylops.waveeqprocessing.AcousticWave2D`).

    The following table shows the current status of the LSM application:

    +------------------+----------------------+-----------+------------+
    |                  |  Kirchhoff integral  |   WKBJ    |   Wave eq  |
    +==================+======================+===========+============+
    | Reflectivity     |          V           |    X      |     X      |
    +------------------+----------------------+-----------+------------+
    | Slowness-squared |          X           |    X      |     V      |
    +------------------+----------------------+-----------+------------+

    Finally, it is worth noting that for both cases the first iteration of an iterative
    scheme aimed at inverting the demigration operator is a simple a projection of the
    recorded data into the model domain. An approximate (band-limited)  image of the subsurface
    is therefore created. This process is referred to in the literature as *migration*.

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
        kind: str = "kirchhoff",
        dottest: bool = False,
        **kwargs_mod,
    ) -> None:
        self.y, self.x, self.z = y, x, z

        if kind == "kirchhoff":
            self.Demop = Kirchhoff(
                z, x, t, srcs, recs, vel, wav, wavcenter, y=y, **kwargs_mod
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
                **kwargs_mod,
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

    def solve(self, d: NDArray, solver: Callable = lsqr, **kwargs_solver):
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
