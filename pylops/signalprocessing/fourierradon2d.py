__all__ = ["FourierRadon2D"]

from typing import Optional, Tuple

import numpy as np
import scipy as sp

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils.backend import get_array_module, get_complex_dtype
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, NDArray

jit_message = deps.numba_import("the radon2d module")
cupy_message = deps.cupy_import("the radon2d module")

if jit_message is None:
    from ._fourierradon2d_numba import _aradon_inner_2d, _radon_inner_2d
if jit_message is None and cupy_message is None:
    from ._fourierradon2d_cuda import _aradon_inner_2d_cuda, _radon_inner_2d_cuda


class FourierRadon2D(LinearOperator):
    r"""2D Fourier Radon transform

    Apply Radon forward (and adjoint) transform using Fast
    Fourier Transform to a 2-dimensional array of size :math:`[n_{p_x} \times n_t]`
    (and :math:`[n_x \times n_t]`).

    Note that forward and adjoint follow the same convention of the time-space
    implementation in :class:`pylops.signalprocessing.Radon2D`.

    Parameters
    ----------
    taxis : :obj:`np.ndarray`
        Time axis
    haxis : :obj:`np.ndarray`
        Spatial axis
    pxaxis : :obj:`np.ndarray`
        Axis of scanning variable :math:`p_x` of parametric curve
    nfft : :obj:`int`
        Number of samples in Fourier transform
    flims : :obj:`tuple`, optional
        Indices of lower and upper limits of Fourier axis to be used in
        the application of the Radon matrix (when ``None``, use entire axis)
    kind : :obj:`str`
        Curve to be used for stacking/spreading (``linear``, ``parabolic``)
    engine : :obj:`str`
        Engine used for computation (``numpy`` or ``numba`` or ``cuda``)
    num_threads_per_blocks : :obj:`tuple`
        Number of threads in each block (only when ``engine=cuda``)
    dtype : :obj:`str`
        Type of elements in input array.
    name : :obj:`str`
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
        If ``engine`` is neither ``numpy``, ``numba``, nor ``cuda``.

    Notes
    -----
    The FourierRadon2D operator applies the Radon transform in the frequency domain.
    After transforming a 2-dimensional array of size
    :math:`[n_x \times n_t]` into the frequency domain, the following linear
    transformation is applied to each frequency component in adjoint mode:

    .. math::
        \begin{bmatrix}
            m(p_{x,1}, \omega_i)  \\
            m(p_{x,2}, \omega_i)  \\
            \vdots          \\
            m(p_{x,N_p}, \omega_i)
        \end{bmatrix}
        =
        \begin{bmatrix}
            e^{-j \omega_i p_{x,1} x^l_1}  & e^{-j \omega_i p_{x,1} x^l_2} &  \ldots & e^{-j \omega_i p_{x,1} x^l_{N_x}}  \\
            e^{-j \omega_i p_{x,2} x^l_1}  & e^{-j \omega_i p_{x,2} x^l_2} &  \ldots & e^{-j \omega_i p_{x,2} x^l_{N_x}}  \\
            \vdots            & \vdots           &  \ddots & \vdots            \\
            e^{-j \omega_i p_{x,N_p} x^l_1}  & e^{-j \omega_i p_{x,N_p} x^l_2} &  \ldots & e^{-j \omega_i p_{x,N_p} x^l_{N_x}}  \\
        \end{bmatrix}
        \begin{bmatrix}
            d(x_1, \omega_i)  \\
            d(x_2, \omega_i)  \\
            \vdots          \\
            d(x_{N_x}, \omega_i)
        \end{bmatrix}

    where :math:`l=1,2`. Similarly the forward mode is implemented by applying the
    transpose and complex conjugate of the above matrix to the model transformed to
    the Fourier domain.

    Refer to [1]_ for more theoretical and implementation details.

    .. [1] Sacchi, M. "Statistical and Transform Methods for
        Geophysical Signal Processing", 2007.

    """

    def __init__(
        self,
        taxis: NDArray,
        haxis: NDArray,
        pxaxis: NDArray,
        nfft: int,
        flims: Optional[Tuple[int, int]] = None,
        kind: str = "linear",
        engine: str = "numpy",
        num_threads_per_blocks: Tuple[int, int] = (32, 32),
        dtype: DTypeLike = "float64",
        name: str = "R",
    ) -> None:
        # engine
        if engine not in ["numpy", "numba", "cuda"]:
            raise NotImplementedError("engine must be numpy or numba or cuda")
        if engine == "numba" and jit_message is not None:
            engine = "numpy"

        # dimensions and super
        dims = len(pxaxis), len(taxis)
        dimsd = len(haxis), len(taxis)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)

        # other input params
        self.taxis, self.haxis = taxis, haxis
        self.nh, self.nt = self.dimsd
        self.px = pxaxis
        self.npx, self.nfft = self.dims[0], nfft
        self.dt = taxis[1] - taxis[0]
        self.dh = haxis[1] - haxis[0]
        self.f = np.fft.rfftfreq(self.nfft, d=self.dt).astype(self.dtype)
        self.nfft2 = len(self.f)
        self.cdtype = get_complex_dtype(dtype)
        self.flims = (0, self.nfft2) if flims is None else flims

        if kind == "parabolic":
            self.haxis = self.haxis**2

        # create additional input parameters for engine=cuda
        if engine == "cuda":
            self.num_threads_per_blocks = num_threads_per_blocks
            (
                num_threads_per_blocks_hpx,
                num_threads_per_blocks_f,
            ) = num_threads_per_blocks
            num_blocks_px = (
                self.dims[0] + num_threads_per_blocks_hpx - 1
            ) // num_threads_per_blocks_hpx
            num_blocks_h = (
                self.dimsd[0] + num_threads_per_blocks_hpx - 1
            ) // num_threads_per_blocks_hpx
            num_blocks_f = (
                self.dims[1] + num_threads_per_blocks_f - 1
            ) // num_threads_per_blocks_f
            self.num_blocks_matvec = (num_blocks_h, num_blocks_f)
            self.num_blocks_rmatvec = (num_blocks_px, num_blocks_f)
        self._register_multiplications(engine)

    def _register_multiplications(self, engine: str) -> None:
        if engine == "numba":
            self._matvec = self._matvec_numba
            self._rmatvec = self._rmatvec_numba
        elif engine == "cuda":
            self._matvec = self._matvec_cuda
            self._rmatvec = self._rmatvec_cuda
        else:
            self._matvec = self._matvec_numpy
            self._rmatvec = self._rmatvec_numpy

    @reshaped
    def _matvec_numpy(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        self.f = ncp.asarray(self.f)
        x = ncp.fft.rfft(x, n=self.nfft, axis=-1)

        H, PX, F = ncp.meshgrid(
            self.haxis, self.px, self.f[self.flims[0] : self.flims[1]], indexing="ij"
        )
        y = ncp.zeros((self.nh, self.nfft2), dtype=self.cdtype)
        y[:, self.flims[0] : self.flims[1]] = ncp.einsum(
            "ijk,jk->ik",
            ncp.exp(-1j * 2 * ncp.pi * F * PX * H),
            x[:, self.flims[0] : self.flims[1]],
        )
        y = ncp.real(ncp.fft.irfft(y, n=self.nfft, axis=-1))[:, : self.nt]
        return y

    @reshaped
    def _rmatvec_numpy(self, y: NDArray) -> NDArray:
        ncp = get_array_module(y)
        self.f = ncp.asarray(self.f)
        y = ncp.fft.rfft(y, n=self.nfft, axis=-1)

        PX, H, F = ncp.meshgrid(
            self.px, self.haxis, self.f[self.flims[0] : self.flims[1]], indexing="ij"
        )
        x = ncp.zeros((self.npx, self.nfft2), dtype=self.cdtype)
        x[:, self.flims[0] : self.flims[1]] = ncp.einsum(
            "ijk,jk->ik",
            ncp.exp(1j * 2 * ncp.pi * F * PX * H),
            y[:, self.flims[0] : self.flims[1]],
        )
        x = ncp.real(ncp.fft.irfft(x, n=self.nfft, axis=-1))[:, : self.nt]
        return x

    @reshaped
    def _matvec_cuda(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros((self.nh, self.nfft2), dtype=self.cdtype)

        x = ncp.fft.rfft(x, n=self.nfft, axis=-1)
        y = _radon_inner_2d_cuda(
            x,
            y,
            ncp.asarray(self.f),
            self.px,
            self.haxis,
            self.flims[0],
            self.flims[1],
            self.npx,
            self.nh,
            num_blocks=self.num_blocks_matvec,
            num_threads_per_blocks=self.num_threads_per_blocks,
        )
        y = ncp.real(ncp.fft.irfft(y, n=self.nfft, axis=-1))[:, : self.nt]
        return y

    @reshaped
    def _rmatvec_cuda(self, y: NDArray) -> NDArray:
        ncp = get_array_module(y)
        x = ncp.zeros((self.npx, self.nfft2), dtype=self.cdtype)

        y = ncp.fft.rfft(y, n=self.nfft, axis=-1)
        x = _aradon_inner_2d_cuda(
            x,
            y,
            ncp.asarray(self.f),
            self.px,
            self.haxis,
            self.flims[0],
            self.flims[1],
            self.npx,
            self.nh,
            num_blocks=self.num_blocks_rmatvec,
            num_threads_per_blocks=self.num_threads_per_blocks,
        )
        x = ncp.real(ncp.fft.irfft(x, n=self.nfft, axis=-1))[:, : self.nt]
        return x

    @reshaped
    def _matvec_numba(self, x: NDArray) -> NDArray:
        y = np.zeros((self.nh, self.nfft2), dtype=self.cdtype)

        x = sp.fft.rfft(x, n=self.nfft, axis=-1)
        _radon_inner_2d(
            x,
            y,
            self.f,
            self.px,
            self.haxis,
            self.flims[0],
            self.flims[1],
            self.npx,
            self.nh,
        )
        y = np.real(sp.fft.irfft(y, n=self.nfft, axis=-1))[:, : self.nt]
        return y

    @reshaped
    def _rmatvec_numba(self, y: NDArray) -> NDArray:
        x = np.zeros((self.npx, self.nfft2), dtype=self.cdtype)

        y = sp.fft.rfft(y, n=self.nfft, axis=-1)
        _aradon_inner_2d(
            x,
            y,
            self.f,
            self.px,
            self.haxis,
            self.flims[0],
            self.flims[1],
            self.npx,
            self.nh,
        )
        x = np.real(sp.fft.irfft(x, n=self.nfft, axis=-1))[:, : self.nt]
        return x
