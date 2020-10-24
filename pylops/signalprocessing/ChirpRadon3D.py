import logging
import numpy as np

from pylops import LinearOperator

try:
    import pyfftw
    from ._ChirpRadon3D import _chirp_radon_3d, _chirp_radon_3d_fftw
except ModuleNotFoundError:
    pyfftw = None
    pyfftw_message = 'Pyfftw not installed, use numpy or run ' \
                     '"pip install pyFFTW" or ' \
                     '"conda install -c conda-forge pyfftw".'

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class ChirpRadon3D(LinearOperator):
    r"""3D Chirp Radon transform

    Apply Radon forward (and adjoint) transform using Fast Fourier Transform
    and Chirp functions to a 3-dimensional array of size
    :math:`[n_x \times n_t]` (and :math:`[n_{x} \times n_t]`).

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
        Two element array :math:`(p_{y,max}, p_{x,max})` of :math:`\tan`
        of maximum stacking angles in :math:`y` and :math:`x` directions
        :math:`(\tan(\alpha_{y,max}), \tan(\alpha_{x,max}))`. If one operates
        in terms of minimum velocity :math:`c_0`, then
        :math:`p_{y.max}=c_0dy/dt` and :math:`p_{x,max}=c_0dx/dt`
    engine : :obj:`str`, optional
        Engine used for fft computation (``numpy`` or ``fftw``)
    dtype : :obj:`str`, optional
        Type of elements in input array.
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
    def __init__(self, taxis, hyaxis, hxaxis, pmax, engine='numpy',
                 dtype='float64', **kwargs_fftw):
        self.dt = taxis[1] - taxis[0]
        self.dy = hyaxis[1] - hyaxis[0]
        self.dx = hxaxis[1] - hxaxis[0]
        self.nt, self.nx, self.ny = taxis.size, hxaxis.size, hyaxis.size
        self.pmax = pmax
        if engine == 'fftw' and pyfftw is not None:
            self.func = lambda x, mode: _chirp_radon_3d_fftw(x, self.dt,
                                                             self.dy, self.dx,
                                                             self.pmax,
                                                             mode=mode,
                                                             **kwargs_fftw)
        elif engine == 'numpy' or (engine == 'fftw' and pyfftw is None):
            self.func = lambda x, mode: _chirp_radon_3d(x, self.dt,
                                                        self.dy, self.dx,
                                                        self.pmax, mode=mode)
        else:
            raise NotImplementedError('engine must be numpy or fftw')
        self.shape = (self.nt * self.nx * self.ny,
                      self.nt * self.nx * self.ny)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        x = x.reshape(self.ny, self.nx, self.nt)
        y = self.func(x, mode='f', )
        return y.ravel()

    def _rmatvec(self, x):
        x = x.reshape(self.ny, self.nx, self.nt)
        y = self.func(x, mode='a')
        return y.ravel()

    def inverse(self, x):
        x = x.reshape(self.ny, self.nx, self.nt)
        y = self.func(x, mode='i')
        return y.ravel()
