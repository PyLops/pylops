import numpy as np


def _chirp_radon_2d(data, dt, dx, pmax, mode='f'):
    r"""2D Chirp Radon transform

    Applies 2D Radon transform using Fast Fourier Transform and Chirp
    functions. (mode='f': forward, 'a': adjoint, and  'i': inverse). See
    Chirp2DRadon operator docstring for more details.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        2D input data of size :math:`[n_x \times n_t]`
    dt : :obj:`float`
        Time sampling :math:`dt`
    dx : :obj:`float`
        Spatial sampling in :math:`x` direction :math:`dx`
    pmax : :obj:`np.ndarray`
        Maximum slope defined as :math:`\tan` of maximum stacking angle
        :math:`x` direction :math:`p_{max} = \tan(\alpha_x_{max})`.
        If one operates in terms of minimum velocity :math:`c_0`, then
        :math:`p_x_{max}=c_0 dy/dt`.
    mode : :obj:`str`, optional
        Mode of operation, 'f': forward, 'a': adjoint, and  'i': inverse

    Returns
    -------
    g : :obj:`np.ndarray`
        2D output of size :math:`[\times n_{x} \times n_t]`

    """
    # define sign for mode
    sign = -1. if mode == 'f' else 1.

    # data size
    (nx, nt) = data.shape

    # find dtype of input
    dtype = np.real(data).dtype
    cdtype = (np.ones(1, dtype=dtype) + 1j * np.ones(1, dtype=dtype)).dtype

    # frequency axis
    omega = (np.fft.fftfreq(nt, 1 / nt) / (nt*dt)).reshape((1, nt)).astype(dtype)

    # slowness sampling
    dp = 2 * dt * pmax / dx / nx

    # spatial axis
    x = (np.fft.fftfreq(2*nx, 1 / (2*nx)) ** 2).reshape((2*nx, 1)).astype(dtype)

    # K coefficients
    K0 = np.exp(sign * np.pi * 1j * dp * dx * omega * x).reshape((2*nx, nt))

    # K conj coefficients
    K = np.conj(np.fft.fftshift(K0, axes=(0,)))[nx//2:3*nx//2, :]

    # perform transform
    h = np.zeros((2 * nx, nt)).astype(cdtype)
    h[0:nx, :] = np.fft.fftn(data, axes=(1,)) * K
    g = np.fft.ifftn(np.fft.fftn(h, axes=(0,)) *
                     np.fft.fftn(K0, axes=(0,)), axes=(0,))
    if mode == 'i':
        g = np.fft.ifftn(g[0:nx, :] * K * abs(omega), axes=(1,)).real * dp * dx
    else:
        g = np.fft.ifftn(g[0:nx, :] * K, axes=(1,)).real
    return g
