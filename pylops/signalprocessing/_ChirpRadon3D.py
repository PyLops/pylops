import numpy as np
import pyfftw


def _chirp_radon_3d(data, dt, dx1, dx2, pmax, mode='f'):
    r"""3D Chirp Radon transform

    Applies 3D Radon transform using Fast Fourier Transform and Chirp
    functions. (mode='f': forward, 'a': adjoint, and  'i': inverse). See
    Chirp3DRadon operator docstring for more details.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        3D input data of size :math:`[n_y \times n_x \times n_t]`
    dt : :obj:`float`
        Time sampling :math:`dt`
    dy : :obj:`float`
        Spatial sampling in :math:`y` direction :math:`dy`
    dx : :obj:`float`
        Spatial sampling in :math:`x` direction :math:`dx`
    pmax : :obj:`np.ndarray`
        Two element array :math:`(p_y_{max}, p_x_{max})` of :math:`\tan`
        of maximum stacking angles in :math:`y` and :math:`x` directions
        :math:`(\tan(\alpha_{y,max}), \tan(\alpha_{x,max}))`. If one operates
        in terms of minimum velocity :math:`c_0`, then
        :math:`p_{y.max}=c_0dy/dt` and :math:`p_{x,max}=c_0dx/dt`
    mode : :obj:`str`, optional
        Mode of operation, 'f': forward, 'a': adjoint, and  'i': inverse

    Returns
    -------
    g : :obj:`np.ndarray`
        3D array of size :math:`[n_{y} \times n_{x} \times n_t]`

    """
    # define sign for mode
    sign = -1. if mode == 'f' else 1.

    # data size
    (ny, nx, nt) = data.shape

    # find dtype of input
    dtype = np.real(data).dtype
    cdtype = (np.ones(1, dtype=dtype) + 1j * np.ones(1, dtype=dtype)).dtype

    # frequency axis
    omega = (np.fft.fftfreq(nt, 1/nt)/(nt*dt)).reshape((1, nt)).astype(dtype)

    # slowness samplings
    dp1 = 2 * dt * pmax[0] / dx1 / nx
    dp2 = 2 * dt * pmax[1] / dx2 / ny

    # spatial axes
    x = (np.fft.fftfreq(2 * nx, 1 / (2 * nx)) ** 2).reshape(
        (1, 2 * nx, 1)).astype(dtype)
    y = (np.fft.fftfreq(2 * ny, 1 / (2 * ny)) ** 2).reshape(
        (2 * ny, 1, 1)).astype(dtype)

    # K coefficients
    K01 = np.exp(sign * np.pi * 1j * dp1 * dx1 * omega * x).reshape(
        (1, int(2 * nx), nt))
    K02 = np.exp(sign * np.pi * 1j * dp2 * dx2 * omega * y).reshape(
        (int(2 * ny), 1, nt))

    # K conj coefficients
    K1 = np.conj(np.fft.fftshift(K01, axes=(1,)))[:, int(nx / 2):int(3 * nx / 2), :]
    K2 = np.conj(np.fft.fftshift(K02, axes=(0,)))[int(ny / 2):int(3 * ny / 2), :, :]

    # perform transform
    h = np.zeros((2 * ny, 2 * nx, nt)).astype(cdtype)
    h[0:ny, 0:nx, :] = np.fft.fftn(data, axes=(2,)) * K1 * K2

    g = np.fft.ifftn(np.fft.fftn(h, axes=(1,)) *
                     np.fft.fftn(K01, axes=(1,)), axes=(1,))
    g = np.fft.ifftn(np.fft.fftn(g, axes=(0,)) *
                     np.fft.fftn(K02, axes=(0,)), axes=(0,))

    if mode == 'i':
        g = np.fft.ifftn(g[0:ny, 0:nx, :] * K1 * K2 * abs(omega) ** 2 *
                         dp1 * dp2 * dx1 * dx2, axes=(2,)).real
    else:
        g = np.fft.ifftn(g[0:ny, 0:nx, :] * K1 * K2, axes=(2,)).real
    return g


def _chirp_radon_3d_fftw(data, dt, dx1, dx2, pmax, mode='f', **kwargs_fftw):
    """3D Chirp Radon transform with pyfftw

    Applies 3D Radon transform using Fast Fourier Transform and Chirp
    functions. (mode='f': forward, 'a': adjoint, and  'i': inverse). See
    Chirp3DRadon operator docstring for more details.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        3D input data of size :math:`[n_y \times n_x \times n_t]`
    dt : :obj:`float`
        Time sampling :math:`dt`
    dy : :obj:`float`
        Spatial sampling in :math:`y` direction :math:`dy`
    dx : :obj:`float`
        Spatial sampling in :math:`x` direction :math:`dx`
    pmax : :obj:`np.ndarray`
        Two element array :math:`(p_y_{max}, p_x_{max})` of :math:`\tan`
        of maximum stacking angles in :math:`y` and :math:`x` directions
        :math:`(\tan(\alpha_{y,max}), \tan(\alpha_{x,max}))`. If one operates
        in terms of minimum velocity :math:`c_0`, then
        :math:`p_{y.max}=c_0dy/dt` and :math:`p_{x,max}=c_0dx/dt`
    mode : :obj:`str`, optional
        Mode of operation, 'f': forward, 'a': adjoint, and  'i': inverse
    **kwargs_fftw : :obj:`int`, optional
        Additional arguments to pass to pyFFTW computations
        (recommended: ``flags=('FFTW_ESTIMATE', ), threads=NTHREADS``)

    """
    # define sign for mode
    sign = -1. if mode == 'f' else 1.

    # data size
    (ny, nx, nt) = data.shape

    # find dtype of input
    dtype = np.real(data).dtype
    cdtype = (np.ones(1, dtype=dtype) + 1j * np.ones(1, dtype=dtype)).dtype

    # frequency axis
    omega = (np.fft.fftfreq(nt, 1 / nt) / (nt * dt)).reshape((1, nt)).astype(dtype)

    # slowness samplings
    dp1 = 2 * dt * pmax[1] / dx1 / nx
    dp2 = 2 * dt * pmax[0] / dx2 / ny

    # pyfftw plans
    data = pyfftw.byte_align(data, n=None, dtype=dtype)
    K1 = pyfftw.empty_aligned((1, nx, nt), dtype=cdtype)
    K2 = pyfftw.empty_aligned((ny, 1, nt), dtype=cdtype)
    K01 = pyfftw.empty_aligned((1, 2 * nx, nt), dtype=cdtype)
    K02 = pyfftw.empty_aligned((2 * ny, 1, nt), dtype=cdtype)
    K01Out = pyfftw.empty_aligned((1, 2 * nx, nt), dtype=cdtype)
    K02Out = pyfftw.empty_aligned((2 * ny, 1, nt), dtype=cdtype)

    hw = pyfftw.zeros_aligned((2 * ny, 2 * nx, nt), dtype=cdtype)
    hw_short = pyfftw.empty_aligned((ny, nx, nt), dtype=cdtype)
    gw = pyfftw.empty_aligned((2 * ny, 2 * nx, nt), dtype=cdtype)
    gw_short = pyfftw.empty_aligned((ny, nx, nt), dtype=cdtype)

    fft_object_t = pyfftw.FFTW(data.astype(cdtype), hw_short, axes=(2,),
                               **kwargs_fftw)
    fft_object_x1 = pyfftw.FFTW(hw, gw, axes=(1,), **kwargs_fftw)
    fft_object_K01w_x1 = pyfftw.FFTW(K01, K01Out, axes=(1,), **kwargs_fftw)
    ifft_object_x1 = pyfftw.FFTW(hw, gw, axes=(1,), direction='FFTW_BACKWARD',
                                 **kwargs_fftw)
    fft_object_x2 = pyfftw.FFTW(gw, hw, axes=(0,), **kwargs_fftw)
    fft_object_K02w_x2 = pyfftw.FFTW(K02, K02Out, axes=(0,), **kwargs_fftw)
    ifft_object_x2 = pyfftw.FFTW(hw, gw, axes=(0,), direction='FFTW_BACKWARD',
                                 **kwargs_fftw)
    ifft_object_t = pyfftw.FFTW(hw_short, gw_short, axes=(2,),
                                direction='FFTW_BACKWARD', **kwargs_fftw)

    # spatial axes
    xw = (np.fft.fftfreq(2 * nx, 1 / (2 * nx)) ** 2).reshape((1, 2 * nx, 1)).astype(dtype)
    yw = (np.fft.fftfreq(2 * ny, 1 / (2 * ny)) ** 2).reshape((2 * ny, 1, 1)).astype(dtype)

    K01[:, :, :] = np.exp(sign*np.pi * 1j * dp1 * dx1 * omega * xw).reshape(
        (1, int(2 * nx), nt))
    K02[:, :, :] = np.exp(sign*np.pi * 1j * dp2 * dx2 * omega * yw).reshape(
        (int(2 * ny), 1, nt))

    K1[:, :, :] = np.conj(np.fft.fftshift(K01, axes=(1,)))[:, int(nx / 2) + 1:int(3 * nx / 2) + 1, :]
    K2[:, :, :] = np.conj(np.fft.fftshift(K02, axes=(0,)))[int(ny / 2) + 1:int(3 * ny / 2) + 1, :, :]

    hw[0:ny, 0:nx, :] = fft_object_t() * K1 * K2
    gw[:, :, :] = ifft_object_x1(fft_object_x1() * fft_object_K01w_x1())
    gw[:, :, :] = ifft_object_x2(fft_object_x2() * fft_object_K02w_x2())

    if mode == 'i':
        g = ifft_object_t(gw[0:ny, 0:nx, :] * K1 * K2 *
                          abs(omega) ** 2 * dp1 * dp2 * dx1 * dx2).real
    else:
        g = ifft_object_t(gw[0:ny, 0:nx, :] * K1 * K2).real
    return g
