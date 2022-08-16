import logging

import numpy as np
import numpy.typing as npt

from pylops import LinearOperator
from pylops.utils.decorators import reshaped

from ._ChirpRadon3D import _chirp_radon_3d

try:
    import pyfftw

    from ._ChirpRadon3D import _chirp_radon_3d_fftw
except ModuleNotFoundError:
    pyfftw = None
    pyfftw_message = (
        "Pyfftw not installed, use numpy or run "
        '"pip install pyFFTW" or '
        '"conda install -c conda-forge pyfftw".'
    )
except Exception as e:
    pyfftw = None
    pyfftw_message = f"Failed to import pyfftw (error:{e}), use numpy."


logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class ChirpRadon3D(LinearOperator):
    r"""3D Chirp Radon transform

    Apply Radon forward (and adjoint) transform using Fast Fourier Transform
    and Chirp functions to a 3-dimensional array of size
    :math:`[n_y \times n_x \times n_t]` (both in forward and adjoint mode).

    Note that forward and adjoint are swapped compared to the time-space
    implementation in :class:`pylops.signalprocessing.Radon3D` and a direct
    `inverse` method is also available for this implementation.

    Parameters
    ----------
    taxis : :obj:`np.ndarray`
        Time axis
    hxaxis : :obj:`np.ndarray`
        Fast patial axis
    hyaxis : :obj:`np.ndarray`
        Slow spatial axis
    pmax : :obj:`np.ndarray`
        Two element array :math:`(p_{y,\text{max}}, p_{x,\text{max}})` of :math:`\tan`
        of maximum stacking angles in :math:`y` and :math:`x` directions
        :math:`(\tan(\alpha_{y,\text{max}}), \tan(\alpha_{x,\text{max}}))`. If one operates
        in terms of minimum velocity :math:`c_0`, then
        :math:`p_{y,\text{max}}=c_0\,\mathrm{d}y/\mathrm{d}t` and :math:`p_{x,\text{max}}=c_0\,\mathrm{d}x/\mathrm{d}t`
    engine : :obj:`str`, optional
        Engine used for fft computation (``numpy`` or ``fftw``)
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)
    **kwargs_fftw
            Arbitrary keyword arguments for :py:class:`pyfftw.FTTW`
            (reccomended: ``flags=('FFTW_ESTIMATE', ), threads=NTHREADS``)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    Refer to [1]_ for the theoretical and implementation details.

    .. [1] Andersson, F and Robertsson J. "Fast :math:`\tau-p` transforms by
        chirp modulation", Geophysics, vol 84, NO.1, pp. A13-A17, 2019.

    """

    def __init__(
        self,
        taxis: npt.ArrayLike,
        hyaxis: npt.ArrayLike,
        hxaxis: npt.ArrayLike,
        pmax: npt.ArrayLike,
        engine: str = "numpy",
        dtype: str = "float64",
        name: str = "C",
        **kwargs_fftw,
    ):
        dims = len(hyaxis), len(hxaxis), len(taxis)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dims, name=name)

        self.ny, self.nx, self.nt = self.dims
        self.dt = taxis[1] - taxis[0]
        self.dy = hyaxis[1] - hyaxis[0]
        self.dx = hxaxis[1] - hxaxis[0]
        self.pmax = pmax
        self.engine = engine
        if self.engine not in ["fftw", "numpy"]:
            raise NotImplementedError("engine must be 'numpy' or 'fftw'")
        self.kwargs_fftw = kwargs_fftw

    @reshaped
    def _matvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        if self.engine == "fftw" and pyfftw is not None:
            return _chirp_radon_3d_fftw(
                x, self.dt, self.dy, self.dx, self.pmax, mode="f", **self.kwargs_fftw
            )
        return _chirp_radon_3d(x, self.dt, self.dy, self.dx, self.pmax, mode="f")

    @reshaped
    def _rmatvec(self, x: npt.ArrayLike) -> npt.ArrayLike:
        if self.engine == "fftw" and pyfftw is not None:
            return _chirp_radon_3d_fftw(
                x, self.dt, self.dy, self.dx, self.pmax, mode="a", **self.kwargs_fftw
            )
        return _chirp_radon_3d(x, self.dt, self.dy, self.dx, self.pmax, mode="a")

    def inverse(self, x: npt.ArrayLike) -> npt.ArrayLike:
        x = x.reshape(self.dimsd)
        if self.engine == "fftw" and pyfftw is not None:
            y = _chirp_radon_3d_fftw(
                x, self.dt, self.dy, self.dx, self.pmax, mode="i", **self.kwargs_fftw
            )
        else:
            y = _chirp_radon_3d(x, self.dt, self.dy, self.dx, self.pmax, mode="i")
        return y.ravel()
