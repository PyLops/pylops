__all__ = ["FourierRadon3D"]

import logging
from typing import Optional, Tuple

import numpy as np
import scipy as sp

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils.backend import get_array_module, get_complex_dtype
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, NDArray

jit_message = deps.numba_import("the radon2d module")

if jit_message is None:
    from ._fourierradon3d_cuda import _aradon_inner_3d_cuda, _radon_inner_3d_cuda
    from ._fourierradon3d_numba import _aradon_inner_3d, _radon_inner_3d

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class FourierRadon3D(LinearOperator):
    r"""3D Fourier Radon transform

    Apply Radon forward (and adjoint) transform using Fast
    Fourier Transform to a 3-dimensional array of size :math:`[n_{p_y} \times n_{p_x} \times n_t]`
    (and :math:`[n_y \times n_x \times n_t]`).

    Note that forward and adjoint follow the same convention of the time-space
    implementation in :class:`pylops.signalprocessing.Radon3D`.

    Parameters
    ----------
    taxis : :obj:`np.ndarray`
        Time axis
    hyaxis : :obj:`np.ndarray`
        Slow spatial axis
    hxaxis : :obj:`np.ndarray`
        Fast spatial axis
    pyaxis : :obj:`np.ndarray`
        Axis of scanning variable :math:`p_y` of parametric curve
    pxaxis : :obj:`np.ndarray`
        Axis of scanning variable :math:`p_x` of parametric curve
    nfft : :obj:`int`
        Number of samples in Fourier transform
    flims : :obj:`tuple`, optional
        Indices of lower and upper limits of Fourier axis to be used in
        the application of the Radon matrix (when ``None``, use entire axis)
    kind : :obj:`tuple`
        Curves to be used for stacking/spreading along the y- and x- axes
        (``("linear", "linear")``, ``("linear", "parabolic")``,
         ``("parabolic", "linear")``, or  ``("parabolic", "parabolic")``)
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
    ValueError
        If ``kind`` is not a tuple of two elements.

    Notes
    -----
    The FourierRadon3D operator applies the Radon transform in the frequency domain.
    After transforming a 3-dimensional array of size
    :math:`[n_y \times n_x \times n_t]` into the frequency domain, the following linear
    transformation is applied to each frequency component in adjoint mode:

    .. math::
        \begin{bmatrix}
            \mathbf{m}(p_{y,1}, \mathbf{p}_{x}, \omega_i)  \\
            \mathbf{m}(p_{y,2}, \mathbf{p}_{x}, \omega_i)  \\
            \vdots          \\
            \mathbf{m}(p_{y,N_{py}}, \mathbf{p}_{x}, \omega_i)
        \end{bmatrix}
        =
        \begin{bmatrix}
            e^{-j \omega_i (p_{y,1} y^l_1 + \mathbf{p}_x \cdot \mathbf{x}^l)}  & e^{-j \omega_i (p_{y,1} y^l_2 + \mathbf{p}_x \cdot \mathbf{x}^l)} &  \ldots & e^{-j \omega_i (p_{y,1} y^l_{N_y} + \mathbf{p}_x \cdot \mathbf{x}^l)}  \\
            e^{-j \omega_i (p_{y,2} y^l_1 + \mathbf{p}_x \cdot \mathbf{x}^l)}  & e^{-j \omega_i (p_{y,2} y^l_2 + \mathbf{p}_x \cdot \mathbf{x}^l)} &  \ldots & e^{-j \omega_i (p_{y,2} y^l_{N_y} + \mathbf{p}_x \cdot \mathbf{x}^l)}  \\
            \vdots            & \vdots           &  \ddots & \vdots            \\
            e^{-j \omega_i (p_{y,N_{py}} y^l_1 + \mathbf{p}_x \cdot \mathbf{x}^l)}  & e^{-j \omega_i (p_{y,N_{py}} y^l_2 + \mathbf{p}_x \cdot \mathbf{x}^l)} &  \ldots & e^{-j \omega_i (p_{y,N_{py}} y^l_{N_y} + \mathbf{p}_x \cdot \mathbf{x}^l)}  \\
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{d}(y_1, \mathbf{x}, \omega_i)  \\
            \mathbf{d}(y_2, \mathbf{x}, \omega_i)  \\
            \vdots          \\
            \mathbf{d}(y_{N_y}, \mathbf{x}, \omega_i)
        \end{bmatrix}

    where :math:`\cdot` represents the element-wise multiplication of two vectors and
    math:`l=1,2`. Similarly the forward mode is implemented by applying the
    transpose and complex conjugate of the above matrix to the model transformed to
    the Fourier domain.

    Refer to [1]_ for more theoretical and implementation details.

    .. [1] Sacchi, M. "Statistical and Transform Methods for
        Geophysical Signal Processing", 2007.
    """

    def __init__(
        self,
        taxis: NDArray,
        hyaxis: NDArray,
        hxaxis: NDArray,
        pyaxis: NDArray,
        pxaxis: NDArray,
        nfft: int,
        flims: Optional[Tuple[int, int]] = None,
        kind: Tuple[str, str] = ("linear", "linear"),
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

        # kind
        if len(kind) != 2:
            raise ValueError("kind must be a tuple of two elements")

        # dimensions and super
        dims = len(pyaxis), len(pxaxis), len(taxis)
        dimsd = len(hyaxis), len(hxaxis), len(taxis)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)

        # other input params
        self.taxis, self.hyaxis, self.hxaxis = taxis, hyaxis, hxaxis
        self.nhy, self.nhx, self.nt = self.dimsd
        self.py, self.px = pyaxis, pxaxis
        self.npy, self.npx, self.nfft = self.dims[0], self.dims[1], nfft
        self.dt = taxis[1] - taxis[0]
        self.dhy = hyaxis[1] - hyaxis[0]
        self.dhx = hxaxis[1] - hxaxis[0]
        self.f = np.fft.rfftfreq(self.nfft, d=self.dt)
        self.nfft2 = len(self.f)
        self.cdtype = get_complex_dtype(dtype)
        self.flims = (0, self.nfft2) if flims is None else flims

        if kind[0] == "parabolic":
            self.hyaxis = self.hyaxis**2
        if kind[1] == "parabolic":
            self.hxaxis = self.hxaxis**2

        # create additional input parameters for engine=cuda
        if engine == "cuda":
            self.num_threads_per_blocks = num_threads_per_blocks
            (
                num_threads_per_blocks_hpy,
                num_threads_per_blocks_hpx,
                num_threads_per_blocks_f,
            ) = num_threads_per_blocks
            num_blocks_py = (
                self.dims[0] + num_threads_per_blocks_hpy - 1
            ) // num_threads_per_blocks_hpx
            num_blocks_px = (
                self.dims[1] + num_threads_per_blocks_hpx - 1
            ) // num_threads_per_blocks_hpx
            num_blocks_hy = (
                self.dimsd[0] + num_threads_per_blocks_hpy - 1
            ) // num_threads_per_blocks_hpx
            num_blocks_hx = (
                self.dimsd[1] + num_threads_per_blocks_hpx - 1
            ) // num_threads_per_blocks_hpx
            num_blocks_f = (
                self.dims[2] + num_threads_per_blocks_f - 1
            ) // num_threads_per_blocks_f
            self.num_blocks_matvec = (num_blocks_hy, num_blocks_hx, num_blocks_f)
            self.num_blocks_rmatvec = (num_blocks_py, num_blocks_px, num_blocks_f)

        self._register_multiplications(engine)

    def _register_multiplications(self, engine: str) -> None:
        if engine == "numba" and jit_message is None:
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
        x = ncp.fft.rfft(x.reshape(-1, self.dims[-1]), n=self.nfft, axis=-1)

        HY, HX = ncp.meshgrid(self.hyaxis, self.hxaxis, indexing="ij")
        PY, PX = ncp.meshgrid(self.py, self.px, indexing="ij")

        HYY, PYY, F = ncp.meshgrid(
            HY.ravel(), PY.ravel(), self.f[self.flims[0] : self.flims[1]], indexing="ij"
        )
        HXX, PXX, _ = ncp.meshgrid(
            HX.ravel(), PX.ravel(), self.f[self.flims[0] : self.flims[1]], indexing="ij"
        )

        y = ncp.zeros((self.nhy * self.nhx, self.nfft2), dtype=self.cdtype)
        y[:, self.flims[0] : self.flims[1]] = ncp.einsum(
            "ijk,jk->ik",
            ncp.exp(-1j * 2 * ncp.pi * F * (PYY * HYY + PXX * HXX)),
            x[:, self.flims[0] : self.flims[1]],
        )
        y = ncp.real(ncp.fft.irfft(y, n=self.nfft, axis=-1))[:, : self.nt]
        return y

    @reshaped
    def _rmatvec_numpy(self, y: NDArray) -> NDArray:
        ncp = get_array_module(y)
        self.f = ncp.asarray(self.f)
        y = ncp.fft.rfft(y.reshape(-1, self.dimsd[-1]), n=self.nfft, axis=-1)

        HY, HX = ncp.meshgrid(self.hyaxis, self.hxaxis, indexing="ij")
        PY, PX = ncp.meshgrid(self.py, self.px, indexing="ij")

        PYY, HYY, F = ncp.meshgrid(
            PY.ravel(), HY.ravel(), self.f[self.flims[0] : self.flims[1]], indexing="ij"
        )
        PXX, HXX, _ = ncp.meshgrid(
            PX.ravel(), HX.ravel(), self.f[self.flims[0] : self.flims[1]], indexing="ij"
        )

        x = ncp.zeros((self.npy * self.npx, self.nfft2), dtype=self.cdtype)
        x[:, self.flims[0] : self.flims[1]] = ncp.einsum(
            "ijk,jk->ik",
            ncp.exp(1j * 2 * ncp.pi * F * (PYY * HYY + PXX * HXX)),
            y[:, self.flims[0] : self.flims[1]],
        )
        x = ncp.real(ncp.fft.irfft(x, n=self.nfft, axis=-1))[:, : self.nt]
        return x

    @reshaped
    def _matvec_cuda(self, x: NDArray) -> NDArray:
        ncp = get_array_module(x)
        y = ncp.zeros((self.nhy, self.nhx, self.nfft2), dtype=self.cdtype)

        x = ncp.fft.rfft(x, n=self.nfft, axis=-1)
        y = _radon_inner_3d_cuda(
            x,
            y,
            ncp.asarray(self.f),
            self.py,
            self.px,
            self.hyaxis,
            self.hxaxis,
            self.flims[0],
            self.flims[1],
            self.npy,
            self.npx,
            self.nhy,
            self.nhx,
            num_blocks=self.num_blocks_matvec,
            num_threads_per_blocks=self.num_threads_per_blocks,
        )
        y = ncp.real(ncp.fft.irfft(y, n=self.nfft, axis=-1))[:, :, : self.nt]
        return y

    @reshaped
    def _rmatvec_cuda(self, y: NDArray) -> NDArray:
        ncp = get_array_module(y)
        x = ncp.zeros((self.npy, self.npx, self.nfft2), dtype=self.cdtype)

        y = ncp.fft.rfft(y, n=self.nfft, axis=-1)
        x = _aradon_inner_3d_cuda(
            x,
            y,
            ncp.asarray(self.f),
            self.py,
            self.px,
            self.hyaxis,
            self.hxaxis,
            self.flims[0],
            self.flims[1],
            self.npy,
            self.npx,
            self.nhy,
            self.nhx,
            num_blocks=self.num_blocks_rmatvec,
            num_threads_per_blocks=self.num_threads_per_blocks,
        )
        x = ncp.real(ncp.fft.irfft(x, n=self.nfft, axis=-1))[:, :, : self.nt]
        return x

    @reshaped
    def _matvec_numba(self, x: NDArray) -> NDArray:
        y = np.zeros((self.nhy, self.nhx, self.nfft2), dtype=self.cdtype)

        x = sp.fft.rfft(x, n=self.nfft, axis=-1)
        _radon_inner_3d(
            x,
            y,
            self.f,
            self.py,
            self.px,
            self.hyaxis,
            self.hxaxis,
            self.flims[0],
            self.flims[1],
            self.npy,
            self.npx,
            self.nhy,
            self.nhx,
        )
        y = np.real(sp.fft.irfft(y, n=self.nfft, axis=-1))[:, :, : self.nt]
        return y

    @reshaped
    def _rmatvec_numba(self, y: NDArray) -> NDArray:
        x = np.zeros((self.npy, self.npx, self.nfft2), dtype=self.cdtype)

        y = sp.fft.rfft(y, n=self.nfft, axis=-1)
        _aradon_inner_3d(
            x,
            y,
            self.f,
            self.py,
            self.px,
            self.hyaxis,
            self.hxaxis,
            self.flims[0],
            self.flims[1],
            self.npy,
            self.npx,
            self.nhy,
            self.nhx,
        )
        x = np.real(sp.fft.irfft(x, n=self.nfft, axis=-1))[:, :, : self.nt]
        return x
